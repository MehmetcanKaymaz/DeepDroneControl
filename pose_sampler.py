import airsimdroneracingvae as airsim
import airsim as airsim2
import numpy as np
import time
from airsimdroneracingvae import Pose, Vector3r, Quaternionr
from scipy.spatial.transform import Rotation as R
import pprint
import math


from TrajGen import trajGenerator, Helix_waypoints
import controller
from Quadrotor import quadrotor


class PoseSampler:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.simLoadLevel('Soccer_Field_Easy')
        self.configureEnvironment()
        quat0 = R.from_euler('ZYX',[-90.,0.,0.],degrees=True).as_quat()
        quat1 = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()
        quat2 = R.from_euler('ZYX',[30.,0.,0.],degrees=True).as_quat()
        quat3 = R.from_euler('ZYX',[45.,0.,0.],degrees=True).as_quat()
        quat4 = R.from_euler('ZYX',[60.,0.,0.],degrees=True).as_quat()
        quat5 = R.from_euler('ZYX',[90.,0.,0.],degrees=True).as_quat()
        self.gate = [Pose(Vector3r(0.,20.,-2.), Quaternionr(quat1[0],quat1[1],quat1[2],quat1[3])),
                     Pose(Vector3r(5.,10.,-2), Quaternionr(quat2[0],quat2[1],quat2[2],quat2[3])),
                     Pose(Vector3r(10.,0.,-1.5), Quaternionr(quat4[0],quat4[1],quat4[2],quat4[3])),
                     Pose(Vector3r(15.,-10.,-3), Quaternionr(quat1[0],quat1[1],quat1[2],quat1[3])),
                     Pose(Vector3r(20.,-20.,-2), Quaternionr(quat4[0],quat4[1],quat4[2],quat4[3]))]
        for i, gate in enumerate(self.gate):
                #print ("gate: ", gate)
                gate_name = "gate_" + str(i)
                self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
                self.client.simSetObjectPose(self.tgt_name, gate, True)

        gate_target = self.gate[0]
        gate_psi = R.from_quat([gate_target.orientation.x_val, gate_target.orientation.y_val, gate_target.orientation.z_val, gate_target.orientation.w_val]).as_euler('ZYX',degrees=False)[0]
        psi_start = gate_psi - np.pi/2 

        self.drone_init = Pose(Vector3r(0.,30.,-2), Quaternionr(quat0[0],quat0[1],quat0[2],quat0[3]))
        quad_pose = [self.drone_init.position.x_val, self.drone_init.position.y_val, self.drone_init.position.z_val, 0., 0., psi_start]

        self.client.simSetVehiclePose(self.QuadPose(quad_pose), True)

        self.client2= airsim2.MultirotorClient()
        self.client2.confirmConnection()
        self.client2.enableApiControl(True)

        self.track=self.gate
        self.fly_to_gate_pid()

    def configureEnvironment(self):
        for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
            self.client.simDestroyObject(gate_object)
            time.sleep(0.05)
    def QuadPose(self,quad_pose):
        x, y, z, roll, pitch, yaw = quad_pose
        q = R.from_euler('ZYX', [yaw, pitch, roll])  # capital letters denote intrinsic rotation (lower case would be extrinsic)
    
        q = q.as_quat()
        t_o_b = Vector3r(x,y,z)
        q_o_b = Quaternionr(q[0], q[1], q[2], q[3])
        return Pose(t_o_b, q_o_b)

    def fly_to_gate_pid(self):
        index=0

        dt=1/200
        T=10
        N=int(T/dt)

        t=np.linspace(0,T,N)

        
        k=0
        while True:
            target=[self.track[index].position.x_val,self.track[index].position.y_val,self.track[index].position.z_val]
            self.client2.moveToPositionAsync(self.track[index].position.x_val,self.track[index].position.y_val,self.track[index].position.z_val,5).join()
            state=self.client.getMultirotorState()
            pose=self.getPosition(state)
            err=self.calculateErr(pose,target)


            if not k==0:
                index+=1

            if index==len(self.track):
                break
            k+=1

            




    def fly_to_gate(self):
        pose=[0,30,-2]
        att=[0,0,np.pi]

        quad=quadrotor.Quadrotor(pose,att)

        
        waypoints = self.get_waypoints()
        ti=time.time()
        traj = trajGenerator(waypoints,max_vel = 8,gamma = 1e6)
        tf=time.time()
        print("delta t:{}".format(tf-ti))
        des_states = traj.get_des_state
        Tmax = traj.TS[-1]

        dt=1/200
        T=10
        N=int(T/dt)

        t=np.linspace(0,T,N)


        for i in range(N):
            state=quad.get_state()
            quadpose=[state[0][0],state[0][1],state[0][2],-state[2][0]/2,-state[2][1]/2,state[2][2]-np.pi/2]
            self.client.simSetVehiclePose(self.QuadPose(quadpose), True)
            time.sleep(.02)
            des_state=des_states(t[i])

            if(t[i] >=Tmax):
                 U, M = controller.run_hover(state, des_state,dt)
            else:
                U, M = controller.run(state, des_state)

            quad.update(dt,U,M)

    def get_waypoints(self):
        x=[]
        y=[]
        z=[]
        x.append(0)
        y.append(30)
        z.append(-2)
        for i in range(len(self.track)):
            x.append(self.track[i].position.x_val)
            y.append(self.track[i].position.y_val)
            z.append(self.track[i].position.z_val)
        return np.stack((x, y, z), axis=-1)
    def getPosition(self,state):
        s = pprint.pformat(state)
        s_arr=s.split("'")

        xs=s_arr[32].split(",")[0].split(":")[1]
        ys=s_arr[34].split(",")[0].split(":")[1]
        zs=s_arr[36].split(",")[0].split(":")[1].split("}")[0]


        x=float(xs)
        y=float(ys)
        z=float(zs)
        return [x,y,z]

    def calculateErr(self,pose1,pose2):
        err2=0
        for i in range(3):
            err2+=pow(pose1[i]-pose2[i],2)
        err=math.sqrt(err2)
        return err




 
