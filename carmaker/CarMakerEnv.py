from pycarmaker import CarMaker, Quantity
from utils import *
import time
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#
class CMEnv():

    def __init__(self, window):

        IP_ADDRESS = "localhost"
        PORT = 16660
        cm_path = "C:\\IPG\\carmaker\\win64-11.0.1\\bin "
        cur_path = "C:\\Users\\user\\Desktop\\graduate_project_0506\\graduate_project_0506\\CarMaker-Environment-for-RL-main\\carmaker "
        os.chdir(cm_path)
        os.system("CM.exe -cmdport 16660 ")
        os.chdir(cur_path)
        self.window = window
        self.alpha = 0.9
        self.signal = 0
        self.steer_signal = 0
        self.velocity_ = 0
        self.state_size = 4*self.window

        self.cm = CarMaker(IP_ADDRESS, PORT)
        self.cm.connect()
        self.cm.load_testrun()
        self.cm.sim_start()
        time.sleep(0.5)
        self.cm.sim_stop()
        print("[INFO] CarMaker Initialized")
        time.sleep(1)

        # Subscribing to CarMaker Vehicle Model Parameters >> IDs are listed in ReferenceManual.pdf
        
        #바꾸기 가능
        
        self.sim_time = Quantity("Time", Quantity.FLOAT)
        self.accel = Quantity("DM.Gas", Quantity.FLOAT)
        self.brake = Quantity("DM.Brake", Quantity.FLOAT)
        self.yaw = Quantity("Car.Yaw", Quantity.FLOAT)
        self.steer = Quantity("DM.Steer.Ang", Quantity.FLOAT)
        self.kp= Quantity("Car.TrackCurv", Quantity.FLOAT)


        #읽기만
        self.Pos_x = Quantity("Car.tx", Quantity.FLOAT)
        self.Pos_y = Quantity("Car.ty", Quantity.FLOAT)
        self.vel_x = Quantity("Car.vx", Quantity.FLOAT)
        self.vel_y = Quantity("Car.vy", Quantity.FLOAT)
        self.Acc = Quantity("Car.ax", Quantity.FLOAT)
        self.tRoad = Quantity("Vhcl.tRoad", Quantity.FLOAT)
        self.Distance = Quantity("Car.Distance", Quantity.FLOAT)
        
        self.cm.subscribe(self.sim_time)
        self.cm.subscribe(self.accel)
        self.cm.subscribe(self.brake)
        self.cm.subscribe(self.yaw)
        self.cm.subscribe(self.steer)
        self.cm.subscribe(self.kp)
        self.cm.subscribe(self.Pos_x)
        self.cm.subscribe(self.Pos_y)
        self.cm.subscribe(self.vel_x)
        self.cm.subscribe(self.vel_y)
        self.cm.subscribe(self.Acc)
        self.cm.subscribe(self.tRoad)
        self.cm.subscribe(self.Distance)
            
       
        # Vehicle Parameters created in Python Env
        
        self.cm.read()
        self.cm.read()
    
    def quantity(self):
        

        
        self.cm.subscribe(self.sim_time)
        self.cm.subscribe(self.accel)
        #self.cm.subscribe(self.brake)
        self.cm.subscribe(self.yaw)
        self.cm.subscribe(self.steer)
        self.cm.subscribe(self.kp)
        self.cm.subscribe(self.Pos_x)
        self.cm.subscribe(self.Pos_y)
        self.cm.subscribe(self.vel_x)
        self.cm.subscribe(self.vel_y)
        self.cm.subscribe(self.Acc)
        self.cm.subscribe(self.tRoad)

            
       
        # Vehicle Parameters created in Python Env
        
        self.cm.read()
        self.cm.read()

    def start(self):
        """
        Start the TestRun
        """
        time.sleep(2)
        self.cm.sim_start()
    
    def stop(self):
        self.cm.sim_stop()
        time.sleep(2)


    def finish(self):
        """
        End the Test Run
        """
        self.cm.sim_stop()
        time.sleep(2)
        os.system('taskkill /IM "' + "HIL.exe" + '" /F')


    def recv_data(self):
        """
        Receive data from CarMaker
        The data must be subscribed at the initialize process
        :return: Received value
        """
        self.cm.read()
        

        ax = self.accel.data
        steer = self.steer.data
        vx = self.vel_x.data
        vy = self.vel_y.data
        yaw = self.yaw.data
        kp = self.kp.data
        tx = self.Pos_x.data
        ty = self.Pos_y.data
        tRoad = self.tRoad.data
        time = self.sim_time.data
        gas = self.accel.data
        brake = self.brake.data
        Distance = self.Distance.data

        return time,ax,steer,vx,vy,yaw,kp,tx,ty, tRoad, gas,brake,Distance

    def reset(self):
        """
        State 및 Score 초기화
        :return: 초기 값
        """
        
        param_curr = [0] * 9
        param_prev = [0] * 2
        
        init_state = [0] *9
        
        init_state = np.reshape(init_state, [1,9])

        score = 0

        return param_curr, init_state, score


    def step(self, param_curr,forced_sig,tRoad_rate, time_step, prev_value, Distance):
        
        info = ""
        isdone = 0

        # param_curr=[dist , vx, vy, yaw, kp, curr_a, curr_delta,tRoad ]
        # parm_past = [prev_a, prev_delta]
        tx = param_curr[0] 
        ty = param_curr[1] 
        vx = param_curr[2] 
        vy = param_curr[3] 
        kp = param_curr[4] 
        steer = param_curr[5] 
        gas = param_curr[6] 
        tRoad = param_curr[7] 
        yaw = param_curr[8]

        # Calculating heading angle error and distance change
        if time_step == 0:
            tx_prev = -231.7
            ty_prev = -2.0
            yaw_prev = 0.09
            kp_prev = 0.02
            Dis_prev = 0.0
        else:
            tx_prev = prev_value[0]
            ty_prev = prev_value[1]
            yaw_prev = prev_value[2]
            kp_prev = prev_value[3]
            Dis_prev = prev_value[4]

        kp_est=(yaw-yaw_prev)/np.sqrt((tx-tx_prev)**2+(ty-ty_prev)**2)
        print(kp_est)
        kp_error=abs((kp_est-kp_prev)/kp_prev*100)
        #print("kp error : %.2f" %(kp_error))

        Dis_change = round(Distance - Dis_prev)

        prev_value[0] = tx
        prev_value[1] = ty
        prev_value[2] = yaw
        prev_value[3] = kp
        prev_value[4] = Distance


        # s_data=[]
        # state = [tx,ty,vx,vy,kp,steer,gas,yaw]
        # #print(state)
        # for i in state:
        #     s_data.append(round(i,5))
        # state=s_data
        tx_=list_min_max([tx],-4100,1300)
        ty_=list_min_max([ty],-2810,500)
        vx_=list_min_max([vx],0,40)
        vy_=list_min_max([vy],-1,3)  
        kp_=list_min_max([kp],-0.08,0.08)
        steer_=list_min_max([steer],-0.7,0.7)
        gas_=list_min_max([gas],0,1)
        yaw_=list_min_max([yaw],-3,3)
        tRoad_ = list_min_max([tRoad],-3.1,3.1)  

        state = tx_+ty_+vx_+vy_+kp_+steer_+gas_+yaw_+tRoad_

        

        # # 한 번 만들어본 구조. 효과 없으면 지우기.
        # # Reward for offset error (steer)
        # if forced_sig[0] == 1:
        #     reward_1 = 0
        # elif abs(tRoad) > 3 or tRoad > -0.1:
        #     reward_1 = 0
        # else:    
        #     reward_1 = -120*(tRoad+0.7)*(tRoad+2.7)

        # #Reward for offset error rate
        # if tRoad_rate == 0:
        #     reward_2 = 0
        # elif tRoad_rate == 1:
        #     reward_2 = 20         

        # # Reward for Gas
        # if forced_sig[1] == 1: #forced brake activated
        #     reward_3 = 0
        # elif forced_sig[1] == 2: #forced accel activated
        #     reward_3 = 0
        # else:
        #     reward_3 = np.clip(-0.12*(vx-20)*(vx-40),0,10)  

        # # Reward for time step 오래 가면 좋으니까
        # reward_4 = 5
        
        w1=0.5
        w2=0.15
        w3=0.25
        w4=0.1

       
        # Reward for offset error (steer)
        if forced_sig[0] == 1:
            reward_1 = 0
        else:    
            #reward_1 = np.exp(-31.25*(tRoad+1.5)*(tRoad+2.0)*1.0)
            #reward_1 = np.clip(-31.25*(tRoad+1.5)*(tRoad+2.0)*1.0, -10, 20)
            reward_1 = w1*(10.)/(1+abs((tRoad+1.7)/0.4)**(2*7.5))

        # Reward for distance 오래 가면 좋으니까
        # reward_2 = w2*10.        
        reward_2 = w2 * Dis_change * (10.)
           
        # Reward for Gas
        if forced_sig[1] == 1: #forced brake
            reward_3 = 0
        elif forced_sig[1] == 2: #forced accel
            reward_3 = 0
        else:
            #reward_3 = np.clip(-0.06*(vx-10)*(vx-22),-30,10)
            reward_3 = w3*(10.)/(1+abs((vx-15)/2.5)**(2*2))

        # Reward for heading angle error
        if kp_error < 15:
            reward_4 = w4*(10.)
        else:
            reward_4 = w4*(-10.)

        '''
        #Reward for offset error rate
        if tRoad_rate == 0:
            reward_4 = w4*(-10.)
        elif tRoad_rate == 1:
            reward_4 = w4*10.
        '''
            
        #print("troad reward:",reward_1)
        #reward_1 = 2*np.exp((-dist/0.2)*(0.5*np.exp(-psi**2/(2*0.005))+2*np.exp(-vx**2/(2*0.2))+1))
        
        # if np.abs(incr_del) < 0.0164:
        #     h_beta_delta = 0
        # else:
        #     h_beta_delta = -30.6*np.abs(incr_del)
        
        # if np.abs(incr_a) < 0.25:
        #     h_beta_a = 0
        # else:
        #     h_beta_a = -2*np.abs(incr_a)

        #print("reward3",reward_3)
          
       # print("v reward:",reward_3)
        TotalReward = reward_1+reward_2+reward_3+reward_4
        # isdone -종료조건
        
        #if abs(tRoad) > 3 or abs(tRoad) < 0.1:#rudfhdlxkf
        if abs(tRoad)>3 or tRoad > -0.1 :  
            isdone = 1
            TotalReward += -5           # 원래 -10
            info = "collision"
        elif abs(yaw) > 1.4: ##헤딩 너무큼
            isdone = 1
            TotalReward += -5
            info = "too big heading angle"
        elif np.sqrt(vx**2 + vy**2) <4: ## 속도 너무 작음
            #print('vx:',vx,'vy:',vy)
            isdone = 2
            TotalReward += -5
            info = "low velocity"
        # elif np.sqrt(vx**2 + vy**2) >15:
        #     isdone = 0 
        #     TotalReward += 3
        # else:
        #     isdone = 0

        next_state = np.reshape(state,[1,9])

        print("R1:%.2f, R2:%.2f, R3:%.2f, R4:%.2f, TR:%.2f, tRoad:%.2f"%(reward_1,reward_2,reward_3,reward_4,TotalReward,tRoad))
        return TotalReward, next_state, isdone, info, prev_value

    def get_signal(self, dx, min_sig=0, max_sig=0.9, inference=True):

        if inference:
            dx = dx
        else:
            dx = dx.numpy()

        self.signal = self.signal + 0.8*dx
        self.signal = np.clip(self.signal, min_sig, max_sig)
        #print("clipped gas:",self.signal)

    def get_steer_signal(self, dx, min_sig=-1.0, max_sig=1.0, inference=True):
        if inference:
            dx = dx
        else:
            dx = dx.numpy()

        self.steer_signal = dx*0.8 # ???????????????????????????????????????????????????????????????????????????????????
        self.steer_signal = np.clip(self.steer_signal, min_sig, max_sig)
        #self.steer_signal=np.random.random_sample()*-1.0
        #print("clipped steer:",self.steer_signal)

    def send_signal(self,forced_sig):
        """
        Send APS, BPS to CarMaker
        :param signal: APS or BPS Signal
        """
        self.signal = round(self.signal, 2)
        self.steer_signal = round(self.steer_signal,2)
        # print(self.signal)
        #print("input signal acc:%.5lf, steer:%.5lf"%(self.signal,self.steer_signal))
        self.cm.DVA_release()
        self.cm.DVA_write(self.brake, 0 )
        
        if forced_sig[0]==0:
            self.cm.DVA_write(self.steer, self.steer_signal)
        if forced_sig[1]==0:
            self.cm.DVA_write(self.accel, self.signal)
        # elif forced_sig[1]==0 and self.signal<0.0:
        #     self.cm.DVA_write(self.accel, 0)
        #     self.cm.DVA_write(self.brake, 1)            


        #self.cm.DVA_write(self.steer, self.steer_signal)

        # elif self.signal == 0:
        #     self.cm.DVA_release()
        #     self.cm.DVA_write(self.accel, self.signal * 0.01)
        #     self.cm.DVA_write(self.brake, self.signal * 0.01)
        #     self.cm.DVA_write(self.steer, self.steer_signal)

        # else:
        #     self.cm.DVA_release()
        #     self.cm.DVA_write(self.brake, round(self.signal * -0.01, 3))
    def forced_accel(self):
        self.cm.DVA_write(self.accel, 0.7)
        self.cm.DVA_write(self.brake, 0)
        self.signal=0.7

    def forced_brake(self):
        self.cm.DVA_write(self.accel, 0)
        self.cm.DVA_write(self.brake, 1)
        self.signal=0.0

    def forced_steer(self,tRoad,Distance):
        
        # if Distance>700:
        #     if tRoad <-2.7:
        #         self.cm.DVA_write(self.steer, 0.4)
        #         self.steer_signal = 0.4

        #     elif tRoad >-0.5:
        #         self.cm.DVA_write(self.steer, -0.4)
        #         self.steer_signal = -0.4 
        if tRoad <-2.8:
                self.cm.DVA_write(self.steer, 0.4)
                self.steer_signal = 0.4

        elif tRoad >-0.3:
            self.cm.DVA_write(self.steer, -0.4)
            self.steer_signal = -0.4 
        





















