from CarMakerEnv import *
from td3_tf2 import *
from utils import gpu_setting
import pandas as pd
import numpy as np
import time

if __name__ == "__main__":
    gpu_setting(3)
    n_episode = 10000
    score_avg = 0
    score_history = []
    n = 0
    drive_cycle = "ftp"
    model_name = "model_cm_ftp_v0.1/td3"
    window = 9
    state_size = 1*window
    
    lat_e=0.0
    lat_e_prev=0.0
    #target_data = pd.read_csv(f"target_data/sim/target_{drive_cycle}_real.csv", names=["target"])
    #target_data = list(target_data["target"])

    cnt = 0
    scores, episode, logger_score, steps, distances = [], [], [],[],[]

    agent = Agent(alpha=0.0001, beta=0.0002, input_dims=(state_size,),
                  tau=0.005, batch_size=32, n_actions=2
                  ,model_chkpt_dir=model_name)
    
    

    for n in range(n_episode):

        env = CMEnv(window)
        time.sleep(1)
       
        param_curr, state, score = env.reset()
        state = state[0]
        
        logger_reward = []

        env.signal = 0
        action = agent.choose_action(state, n)
        env.get_signal(action[0])
        env.get_steer_signal(action[1]) 
        
        prev_X = -233.683
        prev_Y = -2.16724
        
        env.start()
        forced_sig = [0,0]
        env.send_signal(forced_sig)
        time.sleep(1)
        sim_time, ax,steer,vx,vy,yaw,kp,tx,ty,tRoad,gas,brake,Distance = env.recv_data()

        while(1):
            sim_time, ax,steer,vx,vy,yaw,kp,tx,ty,tRoad,gas,brake,Distance = env.recv_data()
            if vx<1.5:
                env.forced_accel()
            else:
                break
        tRoad_rate=0 # 에러가 증가한 상태
        prev_value = [0., 0., 0., 0., 0.] # heading angle error 계산을 위한 새로운 리스트 : 전 스텝의 [tx, ty, yaw, kp, distance]
        for i in range(10000):
            f_signal=False
            forced_sig = [0,0] #[steer,brake]
            time_step=i
            
            sim_time, ax,steer,vx,vy,yaw,kp,tx,ty,tRoad,gas,brake,Distance = env.recv_data()
            lat_e_prev=-1.7-tRoad

            #곡률은 우회전시 마이너스이며 평상시엔 거의 0.001, 작은 곡률은 0.002정도임 
            while(1):
                if yaw<3.14 and yaw>-3.14:
                    break
                yaw=3.14*2+yaw
            
            # if tRoad <-2.8 or tRoad >-0.3:   ###---이거 켜두니까 애가 여기에 의존함. 속도 작게해서 중앙선타고 강제스티어로만 코너링해서 일단 꺼버림 
            #     forced_sig[0]=0
            #     env.forced_steer(tRoad,Distance)
            #     print("forced_steer is on")

            if vx >= 24.0:  # 원래 12였음.
                forced_sig[1] = 1
                env.forced_brake()
                print("forced_brake on")

            if vx <= 4.0:
                forced_sig[1] = 2
                env.forced_accel()
                print("forced_accel on")  # forced_sig[1]=1 or 2로 초기화하고 다시 0으로 안 감.. 이래도 괜찮???????????????????????????????????????????????????

            # if sim_time<2:
            #     continue
            #print("*sim time: %.4f, step: %d" %(sim_time,i+1))
            
            #param_curr=[dist ,vx, vy, yaw, kp, curr_a, curr_delta,tRoad ]
            
            #print(dist)
            param_curr[0] = tx
            param_curr[1] = ty
            param_curr[2] = vx
            param_curr[3] = vy
            param_curr[4] = kp
            param_curr[5] = steer
            param_curr[6] = gas
            param_curr[7] = tRoad
            param_curr[8] = yaw

            #print("time:%.2facc:%.2f,vx:%.2f,vy:%.2f,yaw:%.2f,kp:%.2f,tx:%.2f,ty:%.2f,troad:%.2f"%(sim_time,ax,vx,vy,yaw,kp,tx,ty,tRoad))
            print("v: %.3fkph gas:%.2f, steer:%.2f, brake:%.5f, Distance: %.2f "%(np.sqrt(vx**2 + vy**2)*3.6,gas,steer,brake,Distance))

            #print("[offset: %.3fm /vel:%.3f km/h /steer:%.3f rad" %(tRoad, vx, steer))
            reward, next_state, isdone, info,prev_value = env.step(param_curr,forced_sig,tRoad_rate, time_step, prev_value,Distance) # 현재 reward, state, 종료조건.
            
            next_state = next_state[0] #np.reshape로 [[tx_,ty_,vx_,vy_,kp_,steer_,gas_,yaw_,tRoad_]]꼴로 만들었기 때문에 next_state는 1x9 array가 됨

            
            done = 0
            agent.remember(state, action, reward, next_state, done)
            
            agent.learn()
    
            score += reward

            logger_reward.append(reward)

            state = next_state
            action = agent.choose_action(state, n)

            gas_signal = env.get_signal(action[0])
            steer_signal = env.get_steer_signal(action[1])

            env.send_signal(forced_sig) # 가속도는 그냥 mu받아서 gas, brake로 조절하면되니까 그대로사용.
            #slp = (i / 10) - sim_time 걍 뺐음


            if isdone == 1:
                env.finish()
                print("[INFO]"+info)
                break
            if i >=10:
                if isdone ==2:
                    env.finish()
                    print("[INFO]"+info)
                    break
                    
            if Distance > 990.:
                env.finish()
                print("[INFO] Drive Cycle Complete")
                break
            
            time.sleep(0.01)
            sim_time, ax,steer,vx,vy,yaw,kp,tx,ty,tRoad,gas,brake,Distance = env.recv_data()
            lat_e=-1.7-tRoad
            if abs(lat_e_prev)>abs(lat_e):
                tRoad_rate=1 # 로드에러 감소
            else:
                tRoad_rate=0
        

        cnt+=1
        score_history.append(score)
        score_avg = np.mean(score_history[-20:])
        
        if score_avg > best_score:
            best_score = score_avg
            agent.save_models()

        logger_score.append(score)
        scores.append(score_avg)
        episode.append(cnt)
        steps.append(i+1)
        distances.append(Distance)

        log = "[INFO] episode: {:5d} | ".format(n+1)
        log += "step: {:5d} | ".format(i+1)
        log += "score: {:4.1f} | ".format(score)
        log += "score max: {:4.1f} | ".format(best_score)
        log += "score avg: {:4.1f} | ".format(score_avg)

       # print(log, time.strftime('%x %X'))
        
        # avg_plot = plt.figure()
        # plt.subplot(2,1,1)
        # plt.plot(episode, scores, 'b', label='average_score')
        # plt.plot(episode, logger_score, 'g--', alpha=0.5, label='score')
        # plt.xlabel('episode')
        # plt.ylabel('score')
        # plt.legend()

        # plt.subplot(2,1,2)
        # plt.plot(episode, distances, 'r', label='distances')
        # plt.plot(episode, steps, 'm--', label='steps')
        # plt.legend()
        
        # plt.savefig("TD3_result/save_graph/graph.png")
        # plt.close(avg_plot)

        avg_plot = plt.figure()
        plt.plot(episode, scores, 'b', label='average_score')
        plt.plot(episode, logger_score, 'g--', alpha=0.5, label='score')
        plt.xlabel('episode')
        plt.ylabel('score')
        plt.legend()
        plt.savefig("TD3_result/save_graph/graph.png")
        plt.close(avg_plot)

        dis_step_plot=plt.figure()
        plt.plot(episode, distances, 'r', label='distances')
        plt.plot(episode, steps, 'm--', label='steps')
        plt.xlabel('episode')
        plt.ylabel('distance')
        plt.legend()
        plt.savefig("TD3_result/save_graph/graph_dis.png")
        plt.close(avg_plot)



        try:
            os.mkdir("TD3_result/save_graph/reward")
        except:
            pass

        # if n % 100 == 0:
        #     plot_Status(logger_vel, logger_tg,logger_reward,n)

        log_train = pd.DataFrame([episode, logger_score, scores,distances,steps], index=["episode", "score", "score_avg","distances","step"])
        log_train = log_train.transpose()
        log_train.to_csv('train_results/log_train_{}.csv'.format(model_name.split("/")[0]))
        time.sleep(2)
    env.finish()








# open_CarMaker()
# env = CMEnv()
# time.sleep(1)
# env.cm.sim_start()
# env.send_data(0.)
# _, _,_, status = env.recv_data()
# print(status)
#
#
#
#
# for i, value in enumerate(signal):
#     env.send_data(value)
#     velocity, sim_time,consump, status = env.recv_data()
#
#     # print(status)
#     print(velocity,consump, sim_time, i/10)
#     # print(time.time()-st, sim_time)
#
#     slp = (i/10) - sim_time
#
#     if slp > 0:
#         time.sleep(slp)
#     else:
#         pass
#
#     if status == 0.0:
#         break
#
# env.cm.sim_stop()
