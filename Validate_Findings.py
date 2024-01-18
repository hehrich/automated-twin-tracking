from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *
import os
import numpy as np
import math
import itertools
from ovito.vis import VectorVis
import argparse
def validateFindings(filename,frames_to_compute):
    
    pipeline = import_file(filename)

    def importTables(frame: int, data: DataCollection):
        timestep=data.attributes['Timestep']
        norPlane_table=DataTable(title='ClusterNormalPlanes',identifier='ClNorPlanes',plot_mode=DataTable.PlotMode.NoPlot)

        try:
            with open('Normal_Vectors{}.txt'.format(timestep), 'r') as fobj:
                lines=fobj.readlines()[2:]
            
                a=len(lines)
                norPlane_table.create_property('NorVec',data=np.zeros((a,3)))
                norPlane_table.create_property('ParVec',data=np.zeros((a,3)))
                for row,line in enumerate(lines):
                    temp=(line.split())
                    temp_float=[float(entry) for entry in temp]
                    norPlane_table['NorVec'][row]=temp_float[1:4]
                    norPlane_table['ParVec'][row]=temp_float[4:]
            # print('Import Normal_Vectors{}.txt successful'.format(timestep))
        except FileNotFoundError:
            print('Please change Python script modifier working directory to the location of your .dump file')  
            
        TwinClusterIDs=DataTable(title='TwinClusterIDs',identifier='TwinClusterIDs',plot_mode=DataTable.PlotMode.NoPlot)   
        try:
            with open('TwinClusterIDs{}.txt'.format(timestep), 'r') as fobj:
                lines=fobj.readlines()[2:]
            
                a=len(lines)
                TwinClusterIDs.create_property('ClusterIDs',data=np.zeros((a,2),dtype=int))
            
                for row,line in enumerate(lines):
                    temp=(line.split())
                    
                    temp_int=[int(entry) for entry in temp]
                    
                    TwinClusterIDs['ClusterIDs'][row]=temp_int
            # print('Import TwinClusterIDs{}.txt successful'.format(timestep))
        except FileNotFoundError:
            print('Please change Python script modifier working directory to the location of your .dump file')         
                
        clusters=DataTable(title='Cluster list',identifier='clusters')   
        try:
            with open('Clusters{}.txt'.format(timestep), 'r') as fobj:
                lines=fobj.readlines()[2:]
            
                a=len(lines)
                clusters.create_property('Center of Mass',data=np.zeros((a,3)))
                clusters.create_property('Cluster Size',data=[0]*a)
                for row,line in enumerate(lines):
                    temp=(line.split())
                    
                    temp_int=[float(entry) for entry in temp]
                    
                    clusters['Center of Mass'][row]=temp_int[2:]
                    clusters['Cluster Size'][row]=temp_int[1]
            # print('Import clusters{}.txt successful'.format(timestep))
        except FileNotFoundError:
            print('Please change Python script modifier working directory to the location of your .dump file')  

        global twinids
        twinids=[]
        try:
            with open('twins{}.txt'.format(timestep), 'r') as fobj:
                lines=fobj.readlines()[2:]
                a=len(lines)

                for row,line in enumerate(lines):
                    temp=(line.split())
                    
                    temp_int=[float(entry) for entry in temp]
                    if (temp_int[8:][0])>0: twinids.append(temp_int[0]) #print( )

        except FileNotFoundError:
            print('Please change Python script modifier working directory to the location of your .dump file')     
            
        data.objects.append(norPlane_table)
        data.objects.append(TwinClusterIDs)
        data.objects.append(clusters)
        
    pipeline.modifiers.append(importTables)


    def validate(frame: int, data: DataCollection):
        
        Sel=np.asarray([False]*data.particles.count)
        # Get correct keynames (can be bugged changing from or to headless OVITO)
        part_key_dict=dict.fromkeys(['orientation','position','structure','cluster'])
        clustabl_key_dict=dict.fromkeys(['center','size'])
        def get_keys(dic,actkeys):
            for key in dic.keys():
                for actkey in actkeys:
                    if key in actkey.lower():
                        dic[key]=actkey
            return dic
        
        part_key_dict=get_keys(part_key_dict,list(data.particles.keys()))
        clustabl_key_dict=get_keys(clustabl_key_dict,list(data.tables['clusters'].keys()))
        
        
        COMS=data.tables['clusters'][clustabl_key_dict['center']]
        NorVecs=data.tables['ClNorPlanes']['NorVec']
        Sizes=data.tables['clusters'][clustabl_key_dict['size']]        
        ptwins=data.particles["possibletwingroups"]
        part_or=data.particles[part_key_dict['orientation']]
        part_pos=data.particles[part_key_dict['position']]
        part_strt=data.particles[part_key_dict['structure']]
        part_cls=data.particles[part_key_dict['cluster']]
        vector_data=np.zeros((data.particles.count, 3))
        def NormalizeData(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        twincount=len(data.tables['TwinClusterIDs']['ClusterIDs'])
        def angle(vector1,vector2):
            inner = np.inner(vector1, vector2)
            norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
            cos = inner / norms
            rad = np.arccos(cos)
            deg = np.rad2deg(rad)
            return deg	
        def quaternion_multiply(quaternion1, quaternion0):
            w0, x0, y0, z0 = quaternion0
            w1, x1, y1, z1 = quaternion1
            return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
        
        def QtoR(q):
            w, x, y, z = q
        
            R11 = 1-2*y**2-2*z**2  
            R12 = 2.0*(x*y+w*z)
            R13 = 2.0*(x*z-w*y)
        
            R21 = 2.0*(x*y-w*z)
            R22 = 1-2*x**2-2*z**2      
            R23 = 2.0*(y*z+w*x)
        
            R31 = 2.0*(x*z+w*y)
            R32 = 2.0*(y*z-w*x)
            R33 =  1-2*x**2-2*y**2      
        
            return np.matrix([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])
        
        
        def rot_ax(Q):
            denom = np.sqrt((Q[1,2]-Q[2,1])**2 + (Q[2,0]-Q[0,2])**2 + (Q[0,1]-Q[1,0])**2)
            n1=(Q[1,2]-Q[2,1])/denom
            n2=(Q[2,0]-Q[0,2])/denom
            n3=(Q[0,1]-Q[1,0])/denom
            return(np.array([n1,n2,n3]))

        global orient_x,orient_y,orient_z,orient_w, x_graph, y_graph, debugorientcomp, least_sections,dist_change,seq_pos,seq_ind,valid_list,non_valid_list
        dist_change=[False]*twincount
        not_validated=0
        debugorientcomp=""
        valid_list=[]
        non_valid_list=[]
        def findBestComponent(orient,comp):
            global x_graph,y_graph,debugorientcomp, least_sections,dist_change,seq_pos,seq_ind
            
            lists=sorted(orient.items())
            # x_temp is distance to TB
            # y_temp is orientation
            # pos_temp is current position
            # ind_temp is index of atom
            x_temp,valuetriple=zip(*lists) 
            x=[]
            y=[]
            y_temp,pos_temp,ind_temp=zip(*valuetriple)
            # split for {111} layer clusters 
            splitted_x=np.split(x_temp, np.flatnonzero(np.abs(np.diff(x_temp))>=0.5)+1)
            splitted_y=np.split(y_temp, np.flatnonzero(np.abs(np.diff(x_temp))>=0.5)+1)
        
            splitted_pos=list(np.split(pos_temp, np.flatnonzero(np.abs(np.diff(x_temp))>=0.4)+1))
            splitted_ind=list(np.split(ind_temp, np.flatnonzero(np.abs(np.diff(x_temp))>=0.4)+1))
            y_temp=NormalizeData(y_temp)

            for x_i,y_i in zip(splitted_x,splitted_y):
                x.append(np.median(x_i))
                y.append(np.median(y_i))
            
            if len(y)>1: 
                y_raw=y   
                y=NormalizeData(y)
                
                section_ind=np.flatnonzero(np.abs(np.diff(y))>=0.4)+1
                if section_ind[0]==1:
                    y=np.delete(y,0)
                    x=np.delete(x,0)
                    y_raw=np.delete(y_raw,0)
                    splitted_pos.pop(0)
                    splitted_ind.pop(0)
                    # print("removed left")
                elif section_ind[-1]==len(y)-1:
                    y=np.delete(y,-1)
                    x=np.delete(x,-1)
                    y_raw=np.delete(y_raw,-1)
                    splitted_pos.pop(-1)
                    splitted_ind.pop(-1)
                    # print("removed right")
                
                num_sections=len(np.split(y, np.flatnonzero(np.abs(np.diff(y))>=0.4)+1))
                
                ########## <debug
                b=["_"  if i<0.33 else "-" if 0.34<i<0.66 else "‾" for i in y]
                print("".join(b))
                def minivis(b):
                    e=np.isclose(np.linspace(0,1,10),b,atol=0.1)
                
                    return (str(np.flatnonzero(e)[0]))
                print("".join(list(map(minivis,y))))
                ########## debug>
                
                if (num_sections==5 or num_sections==3) and np.isclose(y[0],y[-1],atol=0.2):
                    
                    if num_sections<least_sections or (num_sections==least_sections and y[0]+y[-1]>1.8) :
                        x_graph,y_graph=x,y 
                        debugorientcomp=comp
                        least_sections=num_sections
                        seq_pos,seq_ind=splitted_pos,splitted_ind

                return np.asarray(y_raw)
                
        for i4,cl in enumerate(data.tables['TwinClusterIDs']['ClusterIDs']):
            
            orient_x,orient_y,orient_z,orient_w = ({} for i in range(4))
            min_cutoff=10
            com1 =COMS[cl[0]-1]
            com2 =COMS[cl[1]-1]
            normvec=NorVecs[cl[0]-1]
            size1=Sizes[cl[0]-1]
            d=np.dot(com1,normvec)
            distance=abs((com2[0]*normvec[0]+com2[1]*normvec[1]+com2[2]*normvec[2]-d)/np.linalg.norm(normvec))
            distance_scaled=distance*2
            verbVec=(com2-com1)/np.linalg.norm(com1-com2)
            finder = CutoffNeighborFinder(min(50,max(0.2*distance,min_cutoff)),data)
            
            
            
            for t in np.arange(-distance_scaled,distance_scaled,min(50,max(0.2*distance,min_cutoff))): 
                s=com1+distance_scaled/2*verbVec+t*verbVec
                asd={}
                for neigh1 in finder.find_at(s): 
                    if part_strt[neigh1.index] == 1 or part_cls[neigh1.index] in cl:
                            
                        
                        punkt=data.particles.position[neigh1.index]
                        d=np.dot(normvec,com1)
                        dist=(punkt[0]*normvec[0]+punkt[1]*normvec[1]+punkt[2]*normvec[2]-d)/np.linalg.norm(normvec) 
                        if np.abs(dist)<15:
                            
                            orient_x[dist]=[part_or[neigh1.index][0],part_pos[neigh1.index],neigh1.index]
                            orient_y[dist]=[part_or[neigh1.index][1],part_pos[neigh1.index],neigh1.index]
                            orient_z[dist]=[part_or[neigh1.index][2],part_pos[neigh1.index],neigh1.index]
                            orient_w[dist]=[part_or[neigh1.index][3],part_pos[neigh1.index],neigh1.index]
                            
                    yield   

            print("Possible twin ",i4+1,cl,"---------------------",end="\n\n")
            x_graph,y_graph=([] for i in range(2))
            least_sections=99
            or_x=findBestComponent(orient_x,"x") 
            or_y=findBestComponent(orient_y,"y") 
            or_z=findBestComponent(orient_z,"z") 
            or_w=findBestComponent(orient_w,"w")  

            if debugorientcomp:     
                or_sig=eval(f'or_{debugorientcomp}')
                a=np.flatnonzero(np.abs(np.diff(NormalizeData(or_sig)))>=0.4)+1
            
            if not np.any(x_graph):
                not_validated+=1
                non_valid_list.append((twinids[i4]))#,i4+1,list(cl)
            else:
                

                out_ind_left=a[0]-1
                out_ind_right=a[-1]
                in_ind_left=a[0]+1
                in_ind_right=a[-1]-2

                left_pos=(list(itertools.chain(*seq_pos[:out_ind_left+1])))
                right_pos=(list(itertools.chain(*seq_pos[out_ind_right-1:])))
                left_ind= (list(itertools.chain(*seq_ind[:out_ind_left+1])))
                right_ind=(list(itertools.chain(*seq_ind[out_ind_right-1:])))
                Sel[right_ind]=True #show only outside fcc regions
                Sel[left_ind]=True  #show only outside fcc regions
                
                ##from left
                q1=[np.median(or_w[:out_ind_left+1]),np.median(or_x[:out_ind_left+1]),np.median(or_y[:out_ind_left+1]),np.median(or_z[:out_ind_left+1])]
                q2=[(or_w[in_ind_left]),-(or_x[in_ind_left]),-(or_y[in_ind_left]),-(or_z[in_ind_left])]

                ##from right
                q1_right=[np.median(or_w[out_ind_right-1:]),np.median(or_x[out_ind_right-1:]),np.median(or_y[out_ind_right-1:]),np.median(or_z[out_ind_right-1:])]
                q2_right=[(or_w[in_ind_right]),-(or_x[in_ind_right]),-(or_y[in_ind_right]),-(or_z[in_ind_right])]


            ################ 
            
                q_rot_left=quaternion_multiply(q2,q1)
                q_rot_right=quaternion_multiply(q1_right,q2_right)
                angle_l=math.degrees(2*np.arctan2(np.linalg.norm(q_rot_left[1:]),q_rot_left[0]))
                angle_r=math.degrees(2*np.arctan2(np.linalg.norm(q_rot_right[1:]),q_rot_right[0]))
                print(f"- angle with two quat - \nleft: {angle_l:.2f} | right: {angle_r:.2f}")            

                Q_rotmat_left=QtoR(q_rot_left)
                angle2_l = math.degrees(np.arccos(0.5*(np.trace(Q_rotmat_left)-1)))
                
                Q_rotmat_right=QtoR(q_rot_right)
                angle2_r = math.degrees(np.arccos(0.5*(np.trace(Q_rotmat_right)-1)))

                q_rot_out = quaternion_multiply(q2_right,q2)
                Q_rotmat_out = QtoR(q_rot_out)
                angle_out = math.degrees(np.arccos(0.5*(np.trace(Q_rotmat_right)-1)))
                            
                print(f"- angle with rot matr -\nleft: {angle2_l:.2f} | right: {angle2_r:.2f}")
                axleft=angle(rot_ax(Q_rotmat_left),normvec)
                axright=angle(rot_ax(Q_rotmat_right),normvec)
                print(f"- rotation axis and degree to normvec -\nleft: {axleft:.2f} | right: {axright:.2f}")
                print(f"= outer regions angle =\nto each other: {angle_out:.2f} | to normvec: {angle(rot_ax(Q_rotmat_out),normvec):.2f}")
                axleft = (axleft-180) if np.isclose(axleft,180,atol=5) else axleft
                axright = (axright-180) if np.isclose(axright,180,atol=5) else axright
                if (np.isclose(angle2_l, 60, atol=0.5) or np.isclose(angle2_r,60, atol=0.5)) and (np.isclose(axleft,0,atol=5) or np.isclose(axright,0,atol=5)):
                    valid_list.append((twinids[i4]))#,i4+1,list(cl)
                else:
                    print(twinids[i4])
                    non_valid_list.append(twinids[i4])
                    # non_valid_list.append((twinids[i4],i4+1,list(cl)))
                print()
            ###################################

        
            table = DataTable(title='Orientation Pair {}; Twingroup {}'.format(cl,i4+1), plot_mode=DataTable.PlotMode.Line,axis_label_x='Normal distance to first TB in angstrom',axis_label_y=f'Normalized Orientaion.{debugorientcomp}')  
            table.x = table.create_property('X coordinates', data=x_graph)
            table.y = table.create_property(f'orientation_{debugorientcomp}', data=y_graph)      
            data.objects.append(table)
            yield i4/twincount
        # TODO: import twin IDs to display actually helpful info
        print(f'Successfully validated {twincount-not_validated} of {twincount} twins.')
        print(f"Validated {valid_list} (ID)",end=" ")
        if non_valid_list:
            print(f', could not validate {non_valid_list} (ID)')
        else:
            print()
    pipeline.modifiers.append(validate)
    print(f"\nAnalyzing {min(frames_to_compute ,pipeline.source.num_frames)} timestep(s)\n")
    for frame in range(min(frames_to_compute ,pipeline.source.num_frames)):
        print(f'\n#######Timestep {frame}#######\n')
        pipeline.compute(frame)
        print('##############')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Python script for automatic twin identification and tracking - Part 2. Place this Python file in the "twinFiles" directory created using Part 1 "identification-and-tracking.py".')
    parser.add_argument('filename',type=str,nargs=1)
    parser.add_argument('--numfram', type=int, default=10000, help='Number of frames to compute after first timestep of provided files.')
    args=parser.parse_args()
    validateFindings(args.filename,args.numfram)
