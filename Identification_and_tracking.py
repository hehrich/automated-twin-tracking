import multiprocessing.managers
import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
import ovito._extensions.pyscript
from ovito.io import import_file, export_file
from ovito.modifiers import (PolyhedralTemplateMatchingModifier,
ExpressionSelectionModifier,
InvertSelectionModifier,
CoordinationAnalysisModifier,
ClusterAnalysisModifier,
ColorCodingModifier,
DeleteSelectedModifier)
from ovito.data import (DataCollection,
CutoffNeighborFinder,
DataTable)
from ovito.vis import Viewport
import numpy as np
from time import perf_counter
from itertools import groupby
from operator import itemgetter
import math
from os import getcwd,mkdir
from os.path import exists
from collections import Counter
import argparse, shutil
from glob import glob

def findAndTrack(filename, size_cutoff, unwrap, frames_to_compute, lattice_type):
    # Data import
    pipeline = import_file(filename)
    print(f"Analyzing {min(frames_to_compute ,pipeline.source.num_frames)} timestep(s)")

    # Structure Identification
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(output_orientation = True))

    # Selection of hcp-like (fcc base lattice) or fcc-like (hcp base lattice) and unidentified structures
    lattyp: str = 2 if lattice_type == "fcc" else 1
    pipeline.modifiers.append(ExpressionSelectionModifier(expression = f'StructureType==0 || StructureType=={lattyp}'))
    # pipeline.modifiers.append(InvertSelectionModifier())

    # Coordination analysis:
    pipeline.modifiers.append(CoordinationAnalysisModifier(
        cutoff = 3.05, 
        only_selected = True))

    # Find possible twin boundary atoms
    def findTBatoms(frame: int, data: DataCollection):
        print(f"Analyzing frame {frame}, timestep {data.attributes['Timestep']}")
        start=perf_counter()
        atom_coord=data.particles['Coordination']
        atom_ptm=data.particles['Structure Type']
        Coord_nonzero=np.flatnonzero(atom_coord)
        Coord_zero=np.flatnonzero(atom_coord==0)
        
        finder=CutoffNeighborFinder(3.05,data)
        Sel=np.asarray([False]*data.particles.count)
        Sel[Coord_zero]=True	
        
        
        
        for i in Coord_nonzero: 
            
            # Sort out atoms that do not satisfy the following conditions
            if atom_coord[i]!=0:
                # Linking atom - two same lattice type, one coordination 6 and no coordination > 6 nearest neighbors
                if atom_ptm[i]==0:
                    neighbors=finder.find(i)
                    ind=[neigh.index for neigh in neighbors if atom_coord[neigh.index]!=0]        		
                    if  not (sum(atom_ptm[ind]==lattyp)>1 and sum(atom_coord[ind]==6)>0) or atom_coord[i]>6:               
                        Sel[i]=True
                # Perfect plane atom - two same lattice type, coordination 6 nearest neighbors
                elif atom_coord[i]==6:
                    neighbors=finder.find(i)
                    ind=[neigh.index for neigh in neighbors if atom_coord[neigh.index]!=0]
                    sum_samestruct=(atom_ptm[ind]==lattyp)
                    sum_coord=(atom_coord[ind]==6)           
                    if not sum(np.all([sum_samestruct,sum_coord],axis=0))>2:    
                        Sel[i]=True
                   
                # Discard atoms with more than 1 neighbor with coordination > 6  
                elif atom_coord[i]>6:
                    neighbors=finder.find(i)
                    ind=[neigh.index for neigh in neighbors if atom_coord[neigh.index]!=0] 
                    if sum(atom_coord[ind]>6)>0: 
                        Sel[i]=True

            # Discard atoms ignored by coordination analysis
            yield
            # prog=int(((i/data.particles.count)*10)+1)*"-"
            # bar="["+prog+((-len(prog)+10)*"_")+"] | "+f"{(((i+1)/data.particles.count)*100):.2f}" + "%"
            # print(f"{bar}",end="\r")
        print()
            
        data.particles_.create_property("Selection",data=Sel)
        stop=perf_counter()
        print(f"Took {stop-start:.2f}s to find {data.particles.count-sum(Sel)} possible TB atoms")
        # debug export for selection evaluation
        # export_file(data,file=f"twinFiles/twins.dump.{frame}",format="lammps/dump",columns =
        #      ["Particle Identifier","Particle Type","Selection", "Structure Type", "Position.X", "Position.Y", "Position.Z","Orientation.X","Orientation.Y","Orientation.Z","Orientation.X","Orientation.Y","Orientation.Z","Orientation.W"],frame=frame)
    pipeline.modifiers.append(findTBatoms)

    # Cluster analysis
    pipeline.modifiers.append(InvertSelectionModifier())
    pipeline.modifiers.append(ClusterAnalysisModifier(
        cutoff = 3.05, 
        only_selected = True, 
        sort_by_size = True, 
        unwrap_particles = unwrap, 
        compute_com = True, 
        cluster_coloring = True))


    # Parameteris planes
    def parameterisePlanes(frame: int, data: DataCollection):
        start=perf_counter()
        
        cluster_table = data.tables['clusters']
        norPlane_table=DataTable(title='Cluster Plane Normal Vectors',
                                identifier='ClNorPlanes',
                                plot_mode=DataTable.PlotMode.NoPlot)

        #Select clusters with more atoms than certain threshold 
        planars = [size for size in cluster_table['Cluster Size'] if size>size_cutoff]
        if not planars:
            print(f"No larger clusters found")
            return
        AtomPosCluster=[] 	

        # Collect data of all atoms belonging to selected clusters
        start = perf_counter()
        clusters=[list(part) for 
                        part in zip(data.particles.position,data.particles['Cluster']) 
                            if 0<part[1]<len(planars)+1]
        clusters.sort(key = itemgetter(1))		
        groups = groupby(clusters, itemgetter(1))	
        for (key, positions) in groups:		
            AtomPosCluster.append([item[0] for item in positions])
        end=perf_counter()
        planar_ind=[]
        evs=[]
        inplaneVec=[]
        
        # Parameterisation of clusters with PCA
        for i2,cluster_pos in enumerate(AtomPosCluster):
            cluster_pos_transposed=(np.transpose(cluster_pos))
            x_=np.mean(cluster_pos_transposed[0])
            y_=np.mean(cluster_pos_transposed[1])
            z_=np.mean(cluster_pos_transposed[2])
            xx,xy,xz,yy,yz,zz=(0,0,0,0,0,0)
            for pos in cluster_pos:
                xx+=(pos[0]-x_)**2
                yy+=(pos[1]-y_)**2
                zz+=(pos[2]-z_)**2
                xy+=(pos[0]-x_)*(pos[1]-y_)
                xz+=(pos[0]-x_)*(pos[2]-z_)
                yz+=(pos[1]-y_)*(pos[2]-z_)
            cova=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
            E,V=np.linalg.eig(cova)
            var_perc=np.array([entry/sum(E)*100 for entry in E])
            size=cluster_table['Cluster Size'][i2]

            # Mark amorphous clusters to prevent from being used in later steps
            if (size<100 and any(var_perc>96)) or not any(var_perc<1) or (size<20 and not any(E<3)):
                planar_ind.append(-1)
                evs.append([0,0,0])
                inplaneVec.append([0,0,0])						
                continue
            ev=V[:,list(E).index(min(E))]
            parVec=V[:,list(E).index(max(E))]
            planar_ind.append(i2)
            evs.append(ev)
            inplaneVec.append(parVec)
        
        # Instantiate tables for later export
        evs_array=np.array(evs)
        inplaneVec_array=np.array(inplaneVec)
        norPlane_table.x = norPlane_table.create_property('ID', data=planar_ind)

        norPlane_table.y = norPlane_table.create_property('X', data=evs_array[:,0])
        norPlane_table.create_property('Y', data=evs_array[:,1])
        norPlane_table.create_property('Z', data=evs_array[:,2])

        norPlane_table.create_property('Xp', data=inplaneVec_array[:,0])
        norPlane_table.create_property('Yp', data=inplaneVec_array[:,1])
        norPlane_table.create_property('Zp', data=inplaneVec_array[:,2])

        data.objects.append(norPlane_table)
        stop=perf_counter()
        print(f"Took {stop-start:.2f}s to parameterise {len(planars)} cluster(s)")
        
        
    pipeline.modifiers.append(parameterisePlanes)


    # Pair parallel planes
    def pairPlanes(frame: int, data: DataCollection):
        start=perf_counter()
        try:
            evIDs=[i for i in data.tables['ClNorPlanes']['ID'] if i >=0]
            evslong=np.array(list(zip(data.tables['ClNorPlanes']['X'],data.tables['ClNorPlanes']['Y'],data.tables['ClNorPlanes']['Z'])))
            evs = evslong[evIDs]
        except KeyError:
            return
        if len(evs)<2:
            print("Less than two planar clusters have been detected")
            return
        cluster_table = data.tables['clusters']
        
        # Anlge calculation between vectors
        def angle(vector1,vector2):
            inner = np.inner(vector1, vector2)
            norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
            cos = inner / norms
            rad = np.arccos(cos)
            deg = np.rad2deg(rad)
            return deg	

        cellDimMax=np.max(data.cell[:,:3])
        norvec=[]	
        parallel_planes_dict={}

        # Combination and comparison of all selected clusters
        for e1,t in enumerate(evs):			
            for e2,other in enumerate(evs):
                i1=evIDs[e1]
                i2=evIDs[e2]
                if i1!=i2: 	
                    layerangle=angle(t,other)
                    com1=cluster_table['Center of Mass'][i1]
                    com2=cluster_table['Center of Mass'][i2]				
                    size1=cluster_table['Cluster Size'][i1]
                    size2=cluster_table['Cluster Size'][i2]
                    dist=np.linalg.norm(com1-com2)
                    d=np.dot(t,com1)
                    norm_dist=abs((com2[0]*t[0]+com2[1]*t[1]+com2[2]*t[2]-d)/np.linalg.norm(t))
                    ggber=angle(com1-com2,t)
                    if layerangle>175:
                        layerangle-=180
                    if ggber>90:	
                        ggber-=180

                    # Conditions for pairing	
                    if  abs(layerangle)<6 and abs(ggber)<90-90*20*norm_dist/cellDimMax and dist<norm_dist*5 and 0.2<size1/size2<5 and norm_dist<40:	
                        temp_dist_list=[math.inf]*4

                        # Pair already found
                        if (i2+1,i1+1) in parallel_planes_dict.items():
                            pass

                        # Pair already paired - compare previous find and replace if needed
                        elif i1+1 in parallel_planes_dict or i2+1 in parallel_planes_dict or i1+1 in parallel_planes_dict.values() or i2+1 in parallel_planes_dict.values():
                            if i1+1 in parallel_planes_dict:
                                temp = cluster_table['Center of Mass'][parallel_planes_dict[i1+1]-1]							
                                temp_dist_list[0] = np.linalg.norm(com1-temp)
                            if i2+1 in parallel_planes_dict:		
                                temp = cluster_table['Center of Mass'][parallel_planes_dict[i2+1]-1]	
                                temp_dist_list[1] = np.linalg.norm(com2-temp)
                            if i1+1 in parallel_planes_dict.values():	
                                ind1=dict((v, k) for k, v in parallel_planes_dict.items())[i1+1]
                                temp = cluster_table['Center of Mass'][ind1-1]							
                                temp_dist_list[2] = np.linalg.norm(com1-temp)							
                            if i2+1 in parallel_planes_dict.values():							
                                ind2=dict((v, k) for k, v in parallel_planes_dict.items())[i2+1]
                                temp = cluster_table['Center of Mass'][ind2-1]						
                                temp_dist_list[3] = np.linalg.norm(com2-temp)

                            ind_min=temp_dist_list.index(min(temp_dist_list))

                            # If no previous match is closer, replace all old matches with current one 
                            if temp_dist_list[ind_min]>dist:
                                if temp_dist_list[0]<math.inf:
                                    parallel_planes_dict.pop(i1+1)							
                                if temp_dist_list[1]<math.inf:
                                    parallel_planes_dict.pop(i2+1)								
                                if temp_dist_list[2]<math.inf:
                                    parallel_planes_dict.pop(ind1)
                                if temp_dist_list[3]<math.inf:
                                    parallel_planes_dict.pop(ind2)								
                                parallel_planes_dict[i1+1]=i2+1

                        # Normal match
                        else:					
                            parallel_planes_dict[i1+1]=i2+1

        # Mark matched twin boundaries
        PossTwins= [0]*data.particles.count			
        pairs=list(parallel_planes_dict.items())
        data.particles_.create_property('PossibleTwinGroups',data=PossTwins)
        indices=[]
        data.apply(ExpressionSelectionModifier(expression='Cluster!=0'))
        tempSel=np.array([data.particles['Selection']==True])
        tempClus=np.array([data.particles['Cluster']])
        onlyCluster=tempClus[tempSel]
        indexAll=np.array([range(data.particles.count)])
        indexSel=indexAll[tempSel]
        temp_pairs=list(sum(pairs,()))
        for index,at in zip(indexSel,onlyCluster):	
            if at in temp_pairs:
                indices.append([index,at])
        for nr,tupl in enumerate(pairs):
            for indx in indices:
                if indx[1] in tupl: 
                    data.particles['PossibleTwinGroups'][indx[0]]=nr+1

        # Data tables for export
        TwinClusterIDs=DataTable(title='TwinClusterIDs',identifier='TwinClusterIDs',plot_mode=DataTable.PlotMode.NoPlot)
        TwinClusterIDs.x=TwinClusterIDs.create_property('p1',data=list(parallel_planes_dict.keys()))
        TwinClusterIDs.y=TwinClusterIDs.create_property('p2',data=list(parallel_planes_dict.values()))
        data.objects.append(TwinClusterIDs)
        stop=perf_counter()
        print(f"Paired {len(pairs)} pair(s) in {stop-start:.2f}s")
        

    pipeline.modifiers.append(pairPlanes)
    
    # TODO: Need to implement case of no detected twins at timestep 0
    #       as 'if frame==0' will stop working as intended!! (some indicator like first twins found) 
    
    # Track twins
    def trackTwins(frame: int, data: DataCollection):
        start=perf_counter()
        global path
        path = getcwd()
        timestep_curr = data.attributes['Timestep']
        if not exists(path+"/twinFiles"): mkdir(path+"/twinFiles")
        try:
            pairs = list(zip(data.tables["TwinClusterIDs"]["p1"],data.tables["TwinClusterIDs"]["p2"]))
            assert len(pairs) != 0
        except KeyError:
            print(f"No pairs have been found at timestep {timestep_curr} (TwinClusterID table not found)\n")
            timestep_pre = pipeline.source.compute(frame-1).attributes['Timestep']
            if len(glob("twinFiles/twins*.txt"))!=0:
                shutil.copy(f"twinFiles/twins{timestep_pre}.txt", f"twinFiles/twins{timestep_curr}.txt")
            return
        except AssertionError:
            print(f"No pairs have been found at timestep {timestep_curr} (TwinClusterID table is empty)\n")
            timestep_pre = pipeline.source.compute(frame-1).attributes['Timestep']
            if len(glob("twinFiles/twins*.txt"))!=0:
                shutil.copy(f"twinFiles/twins{timestep_pre}.txt", f"twinFiles/twins{timestep_curr}.txt")
            # export data for visualization anyway
            data.particles_.create_property('TwinID',data=[0]*data.particles.count)
            # twin info not needed when no twin found
            export_file(data,file="{}/twinFiles/twins.dump.{}".format(path,timestep_curr),format="lammps/dump",columns =
             ["Particle Identifier","Particle Type", "Structure Type", "Position.X", "Position.Y", "Position.Z","Orientation.X","Orientation.Y","Orientation.Z","Orientation.W"],frame=frame)
            
            return
        planes=list(zip(data.tables["ClNorPlanes"]["X"],data.tables["ClNorPlanes"]["Y"],data.tables["ClNorPlanes"]["Z"]))
        atom_cluster=data.tables["clusters"]
        atom_posstwin=data.particles["PossibleTwinGroups"]
        twins_list=[]

        def angle(vector1,vector2):
            inner = np.inner(vector1, vector2)
            norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
            cos = inner / norms
            rad = np.arccos(cos)
            deg = np.rad2deg(rad)
            return deg	

        # On frame 0 no tracking can be performed but reference for next timestep is created
        if frame == 0 or len(glob("twinFiles/twins*.txt"))==0:
            for i,e in enumerate(pairs):
                yield
                p1=pairs[i][0]
                p2=pairs[i][1]
                v1=planes[p1-1]
                v_mean=planes[p2-1]
                COM1=atom_cluster["Center of Mass"][p1-1]
                COM2=atom_cluster["Center of Mass"][p2-1]
                COM_mean=(COM1+COM2)/2
                twins_list.append([v_mean,COM_mean])
            # Create data tables for tracking         
            twins_array=(np.asarray(twins_list))
            data.particles_.create_property('TwinID',data=atom_posstwin)
            twins=DataTable(title='Twins tracked at {}'.format(timestep_curr),identifier='twins',plot_mode=DataTable.PlotMode.NoPlot)
            twins.x=twins.create_property('id',data=range(1,i+2))
            twins.y=twins.create_property('COM_mean',data=twins_array[:,1])
            twins.create_property('v_mean',data=twins_array[:,0])
            twins.create_property('origin',data=[timestep_curr]*(i+1))
            twins.create_property('find_nmbr',data=range(1,i+2))
            data.objects.append(twins)    
            export_file(data.tables['twins'],file="{}/twinFiles/twins{}.txt".format(path,timestep_curr),format="txt/table",frame=frame)
            # twindvalidation file needs all ptm info
            export_file(data,file=f"twinFiles/twins.dump.{timestep_curr}",format="lammps/dump",columns =
             ["Particle Identifier","Particle Type", "Structure Type", "Position.X", "Position.Y", "Position.Z","Orientation.X","Orientation.Y","Orientation.Z","Orientation.W","Cluster","TwinID"],frame=frame)
    
            export_file(data.tables['ClNorPlanes'],file="{}/twinFiles/Normal_Vectors{}.txt".format(path,timestep_curr),format="txt/table")
            export_file(data.tables['TwinClusterIDs'],file="{}/twinFiles/TwinClusterIDs{}.txt".format(path,timestep_curr),format="txt/table")	
            export_file(data.tables['clusters'],file="{}/twinFiles/Clusters{}.txt".format(path,timestep_curr),format="txt/table")
            data.apply(ColorCodingModifier(property="TwinID",gradient=ColorCodingModifier.Viridis()))

        # Tracking of twins after first frame
        else:  
            timestep_pre=pipeline.source.compute(frame-1).attributes['Timestep']
            pairs_list=[]

            # Import reference twin configuration of previous timestep
            try:
                with open('twinFiles/twins{}.txt'.format(timestep_pre), 'r') as fobj:
                    lines=fobj.readlines()[2:]
                    COM_mean_old=[]
                    v_mean_old=[]
                    origins_old=[]
                    a=len(lines)                    
                    for row,line in enumerate(lines):
                        temp=(line.split())  
                        temp_float=[float(entry) for entry in temp]
                        origins_old.append(int(temp_float[7]))
                        if sum(temp_float[4:7]) !=0:
                            COM_mean_old.append(temp_float[1:4])
                            v_mean_old.append(temp_float[4:7])                      
                        else:
                            COM_mean_old.append("lost")
                            v_mean_old.append(temp_float[1])              
                    # print('Import twins{}.txt successful'.format(timestep_pre))
            except FileNotFoundError:
                print('Please change Python script modifier working directory to the location of your .dump file')   

            twin_tracking={}
            old_twins=[False]*len(COM_mean_old)
            olds=[*(zip(COM_mean_old,v_mean_old))]
            num_twins=len(olds)

            for i,e in enumerate(pairs):
                p1=pairs[i][0]
                p2=pairs[i][1]
                v1=planes[p1-1]
                v_mean=planes[p2-1]
                COM1=atom_cluster["Center of Mass"][p1-1]
                COM2=atom_cluster["Center of Mass"][p2-1]
                COM_mean=(COM1+COM2)/2
                new=True         
                pairs_list.append([v_mean,COM_mean])
                pairs_array=(np.asarray(pairs_list))
                
                for w,i1 in enumerate(olds):
                    yield
                    if not isinstance(i1[0],str):                    
                        d=np.dot(i1[1],i1[0])
                        norm_dist_tr=abs((COM_mean[0]*i1[1][0]+COM_mean[1]*i1[1][1]+COM_mean[2]*i1[1][2]-d)/np.linalg.norm(i1[1]))
                        # Conditions for tracking twin
                        if  norm_dist_tr < 15 and np.linalg.norm(COM_mean-i1[0])<50 and (angle(v_mean,i1[1])<7 or angle(v_mean,i1[1])>173):                                                                                                                                   
                            old_twins[w]=True
                            new=False
                            twin_tracking[i+1]=w+1
                            break

                # If no match was found add twin to end of list   
                if new:
                    num_twins+=1
                    twin_tracking[i+1]=-num_twins

            # Information of not tracked twins will be carried over
            if np.any(np.invert(old_twins)):
                ind_lost=(np.flatnonzero(np.invert(old_twins))) 
                for ind in ind_lost:
                    if isinstance(olds[ind][0],str):
                        twin_tracking[-(ind+1)]=-olds[ind][1]
                    else:
                        twin_tracking[-(ind+1)]=ind+1

            # Duplicate pairings are resolved by selecting closest match                 
            duplicates=(list(filter(lambda x : x > 0,(Counter(twin_tracking.values()) - Counter(set(twin_tracking.values()))).elements())))
            to_compare={}        
            for key,value in twin_tracking.items():
                if value in duplicates:
                    if value in to_compare:
                        to_compare[value].append(key)
                    else:
                        to_compare[value]=[key]

            for key1, value1 in to_compare.items():

                def norm(x): # dont define this function in the loop
                    d=np.dot(COM_mean_old[key1-1],v_mean_old[key1-1])
                    norm_dist_tr=abs((x[0]*v_mean_old[key1-1][0]+x[1]*v_mean_old[key1-1][1]+x[2]*v_mean_old[key1-1][2]-d)/np.linalg.norm(v_mean_old[key1-1]))
                    return norm_dist_tr 
               
                dist_old=list(map(norm,pairs_array[:,1][np.asarray(value1)-1]))
                value1.pop(dist_old.index(min(dist_old)))
                
                for left in value1:
                    twin_tracking.pop(left)
                    num_twins+=1               
                    twin_tracking[left]=-num_twins

            twins_list=[0]*(num_twins)
            origins=[0]*num_twins
            
            # Updating data table of all twins including tracked, new and lost twins
            for key in twin_tracking:            
                if key<0:
                    # Twin already lost
                    if twin_tracking[key]<0:
                        twins_list[-key-1]=np.asarray([[0,0,0],[-twin_tracking[key],0,0]])     
                    # Twin lost this timestep              
                    else:
                        twins_list[-key-1]=np.asarray([[0,0,0],[timestep_curr,0,0]])
                    origins[-key-1]=origins_old[-key-1]
                else:

                    #New Twin
                    if twin_tracking[key]<0:
                        twins_list[-twin_tracking[key]-1]=pairs_list[key-1]
                        origins[-twin_tracking[key]-1]=timestep_curr
                    #Twin with ID higher than len(pairs) meaning we have fewer twins than before but still find matches
                    elif key<=len(pairs) and twin_tracking[key]>len(pairs):
                        twins_list[twin_tracking[key]-1]=pairs_list[key-1]
                        origins[twin_tracking[key]-1]=origins_old[twin_tracking[key]-1]
                    #Twin normally tracked
                    else:
                        twins_list[twin_tracking[key]-1]=pairs_list[key-1]
                        origins[twin_tracking[key]-1]=origins_old[twin_tracking[key]-1]

            # Assign twin property
            twins_array=(np.asarray(twins_list))
            TwinID= [0]*data.particles.count
            data.particles_.create_property('TwinID',data=TwinID)
            indices=[] 
            data.apply(ExpressionSelectionModifier(expression='PossibleTwinGroups!=0'))
            tempSel=np.array([data.particles['Selection']==True])
            tempPoss=np.array([data.particles['PossibleTwinGroups']])
            onlyPoss=tempPoss[tempSel]
            indexAll=np.array([range(data.particles.count)])
            indexSel=indexAll[tempSel]
            for index,at in zip(indexSel,onlyPoss):
                for key in twin_tracking.keys():
                    if key == at:
                        if twin_tracking[key]<0:
                            data.particles["TwinID"][index]=-twin_tracking[key]
                        else:
                            data.particles["TwinID"][index]=twin_tracking[key] 

            # Export data
            twins=DataTable(title='Twins tracked at {}'.format(timestep_curr),identifier='twins',plot_mode=DataTable.PlotMode.NoPlot)
            twins.x=twins.create_property('id',data=range(1,num_twins+1))
            twins.y=twins.create_property('COM_mean',data=twins_array[:,1])
            twins.create_property('v_mean',data=twins_array[:,0])
            twins.create_property('origin',data=origins)
 
            tr=[0]*num_twins
            
            for findid,twid in twin_tracking.items():
                if findid < 0 and twid < 0:
                    tr[-findid-1] = 0
                elif findid < 0:
                    tr[twid-1] = 0
                elif twid < 0:
                    tr[-twid-1] = findid
                else:
                    tr[twid-1] = findid

            twins.create_property('find_nmbr',data=tr)
                                                                   
            data.objects.append(twins)
            export_file(data.tables['twins'],file="{}/twinFiles/twins{}.txt".format(path,timestep_curr),format="txt/table",frame=frame)
            export_file(data,file=f"twinFiles/twins.dump.{timestep_curr}",format="lammps/dump",columns =
             ["Particle Identifier","Particle Type", "Structure Type", "Position.X", "Position.Y", "Position.Z","Orientation.X","Orientation.Y","Orientation.Z","Orientation.W","Cluster","TwinID"],frame=frame)
            export_file(data.tables['ClNorPlanes'],file="{}/twinFiles/Normal_Vectors{}.txt".format(path,timestep_curr),format="txt/table")
            export_file(data.tables['TwinClusterIDs'],file="{}/twinFiles/TwinClusterIDs{}.txt".format(path,timestep_curr),format="txt/table")	
            export_file(data.tables['clusters'],file="{}/twinFiles/Clusters{}.txt".format(path,timestep_curr),format="txt/table")
            data.apply(ColorCodingModifier(property="TwinID",gradient=ColorCodingModifier.Viridis(),end_value=num_twins))
        stop=perf_counter()
        print(f"Took {stop-start:.2f}s to track twins")

            
    pipeline.modifiers.append(trackTwins)


    for frame in range(min(frames_to_compute ,pipeline.source.num_frames)):
        pipeline.compute(frame)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python script for automatic twin identification and tracking - Part 1. Place this Python file in the directory of the atom data files of interest.')
    parser.add_argument('filename',type=str,nargs=1)
    parser.add_argument("--sizcut", type=int, default=100,help='Change clustersize cutoff to include or exclude smaller clusters (Default is 100 atoms).')
    parser.add_argument('--unwrap',action=argparse.BooleanOptionalAction,help='In case of periodic boundaries in the simulation, this should be set to True.')
    parser.add_argument('--numfram', type=int, default=10000, help='Number of frames to compute after first timestep of provided files.')
    parser.add_argument('--lattice_type', '-L',type=str,nargs=1)
    
    args=parser.parse_args()
    findAndTrack(args.filename,args.sizcut,args.unwrap,args.numfram,args.lattice_type)
