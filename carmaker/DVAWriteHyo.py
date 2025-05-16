from pycarmaker import CarMaker, Quantity
import time

# 1 - Initialize pyCarMaker
IP_ADDRESS = "localhost"
PORT = 16660
cm = CarMaker(IP_ADDRESS, PORT)

# 2 - Connect to CarMaker
cm.connect()
time.sleep(1)


# 3 - Create a Quantity

#바꾸기 가능
sim_status = Quantity("SimStatus", Quantity.INT, True)
sim_time = Quantity("Time", Quantity.FLOAT)
accel = Quantity("DM.gas", Quantity.FLOAT)
brake = Quantity("DM.brake", Quantity.FLOAT)
yaw = Quantity("Car.yaw", Quantity.FLOAT)
whlAngFL = Quantity("DM.Steer.Ang", Quantity.FLOAT)

#읽기만
Pos_x = Quantity("Car.tx", Quantity.FLOAT)
Pos_y = Quantity("Car.ty", Quantity.FLOAT)
vel_x = Quantity("Car.vx", Quantity.FLOAT)
vel_y = Quantity("Car.vy", Quantity.FLOAT)

Acc = Quantity("Car.ax", Quantity.FLOAT)


cm.subscribe(sim_status)
cm.subscribe(sim_time)
cm.subscribe(accel)
cm.subscribe(brake)
cm.subscribe(Pos_x)
cm.subscribe(Pos_y)
cm.subscribe(vel_x)
cm.subscribe(vel_y)
cm.subscribe(whlAngFL)
cm.subscribe(Acc)
cm.subscribe(yaw)

cm.read()
cm.read()
print(sim_status.data)
print("WAITING...")

for i in range(3):
    cnt = 0
    print("EPISDE ",i)
    cm.sim_start()
    while True:
        cm.read()
        print(vel_x.data, sim_time.data)
        cm.DVA_write(accel, 0.5)
        time.sleep(0.5)


        cm.DVA_write(accel, 0)
        cm.DVA_release()

        cm.DVA_write(brake, 0.5)
        time.sleep(0.5)
        cm.DVA_write(brake, 0)
        cm.DVA_release()

        if cnt > 5:
            cm.sim_stop()
            print("SIM STATS ", sim_status.data)
            break

        cnt+=1

