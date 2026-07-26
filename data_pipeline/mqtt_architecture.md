## MQTT Basic
Message Queue Telemetry Transport Protocol (MQTT)

MQTT is a simple, lightweight messaging protocol used to establish communication between multiple devices. 

It is a TCP-based protocol (message based protocol) relying on the **publish-subscribe model.** 
The key component in MQTT is the **MQTT broker**. 

The main task of MQTT broker is dispatching messages to the clients (“subscribers”). 

In other words, it receives messages from publisher and dispatches these messages to the subscribers. 
While it dispatches messages, the MQTT broker uses the **topic to filter the clients** that will receive the message. 
The topic is a string and it is possible to combine the topics creating topic levels.

<img width="757" height="578" alt="image" src="https://github.com/user-attachments/assets/fbdb4062-eac4-4bb2-8549-1c23dda2e9dc" />

## MQTT installation on windows
```
Link:  https://mosquitto.org/download/
```

```
**Windows**
- mosquitto-2.1.2-install-windows-x64.exe
- mosquitto-2.1.2-install-windows-x86.exe
```
Older installers can be found at https://mosquitto.org/files/binary/.

See also README-windows.md after installing.

## MQTT start

# Step 1:Check if the Mosquitto service exists

Open PowerShell as Administrator and run:
```
**sc query mosquitto**
```
If you see:
```
**STATE              : 4 RUNNING**
```
the broker is running.

If you see:
```
**STATE              : 1 STOPPED**
```
```
MQTT version :paho-mqtt==2.1.0
 
BROKER = "localhost"

PORT = 1883
```

Start the Mosquitto broker
If installed as a Windows service
```
Run:

net start mosquitto
```
or open Services (services.msc) and start the Mosquitto Broker service.

# Step 2: If the service doesn't exist

Go to the Mosquitto installation folder (typically):
```
C:\Program Files\mosquitto
```
Open a Command Prompt in that folder and run:
```
mosquitto.exe -v
```
You should see:
```
mosquitto version 2.x starting
Opening ipv4 listen socket on port 1883.
Opening ipv6 listen socket on port 1883.
```
Keep this window open. Your simulator connects to this running broker.

# Step 3: Test the broker

Open another terminal and run:
```
mosquitto_sub -h localhost -t test
```
Open a third terminal:
```
mosquitto_pub -h localhost -t test -m "Hello"
```
If the subscriber prints:

Hello

your broker is working correctly.

# Step 4: Verify port 1883

Run:
```
netstat -ano | findstr :1883
```
You should see:
```
TCP    0.0.0.0:1883    LISTENING
```
If you don't, the broker is not running.

Step 5: Run your simulator
```
python simulator.py --anomaly none
```

 <img width="707" height="273" alt="image" src="https://github.com/user-attachments/assets/71a79f3a-7e71-47cf-899d-496aa42b4349" />

## Quality of Service (QoS)

Quality of Service (QoS) in MQTT messaging is an agreement between sender and receiver on the guarantee of delivering a message.

There are three levels of QoS:

**0 at most once
1 at least once
2 exactly once**

## our project clients
1. Tempearture
2. vibration
3. Door event


  

