#include <Wire.h>
#include <ServoTimer2.h>// can't use servo.h because it has a conflict with virtualwire
#include <VirtualWire.h> //for nrf replacer RF module 

//motor driver pin declaring
#define leftMotorEnable 44 
#define leftMotorForward 29
#define leftMotorBackward 31
#define rightMotorEnable 45
#define rightMotorForward 28
#define rightMotorBackward 30

// ultrasonic sensor pin declaring
#define ultraSonicEcho 12
#define ultraSonicTrig 11

//Infared sensor pin declaring
#define irSensorBack A2
#define irSensorLeft A3
#define irSensorRight A4
#define irSensorHeadRight A0 
#define irSensorHeadLeft A1

//servo motor pin declaring
#define servoEye 2
#define servoClaw 3
#define servoFrontBack 4
#define servoLeftRight 5
#define servoUpDown 6

//define bluetooth Control mode button
#define bluetoothMode 22
//define voice Control mode button
#define voiceMode 23

//for simulation only
const int receive_pin = 13;
String message; 
String myString;

//some parameters that we can adjust based on our needs
char voice =' ';//creating varibale to store bluetooth command
String Voice ="";//creating varibale to store voice command
int leftRightTurningDelay = 400; //adjust the value for fully turning
int leftSeeAngle = 170; // adjust the value of left turning ultrasonic sensor using servo motor 
int rightSeeAngle = 10;// adjust the value of right turning ultrasonic sensor using servo motor 
int forwardSeeAngle = 90; //adjust the forward looking value if you need to
int leftRightSeeDelay = 700; //adjust the value for correct mesurement of distance
int maxSpeed = 255; // value should lie between 0 - 255
int minSpeed = 70; //the value at motor should turn on in gesture control mode. 
//robotic claw initial position in terms of angle value 0- 90. change the value according to your arm initial position
int frontBackMovement = 90;
int leftRightMovement = 90;
int clawMovement = 90;
int upDownMovement = 90;
int armStepSize = 5;
bool positionChange = false;

//creating a structure to store all the data we received in one package
struct data {
  int xAxis;
  int yAxis;
  float felxSensorClawValue;
  bool clawInterrputValue;
  int modeInterruptValue;
};

//servo motor objects
ServoTimer2 servoEyeControl; 
ServoTimer2 servoClawControl; 
ServoTimer2 servoFrontBackControl; 
ServoTimer2 servoLeftRightControl; 
ServoTimer2 servoUpDownControl;
data receivedData; // creating a data object to store the received data in that object


void setup() {
  //setting motor driver connected pins as output
  pinMode(leftMotorEnable,OUTPUT);
  pinMode(leftMotorForward,OUTPUT);
  pinMode(leftMotorBackward,OUTPUT);
  pinMode(rightMotorEnable,OUTPUT);
  pinMode(rightMotorForward,OUTPUT);
  pinMode(rightMotorBackward,OUTPUT);

  //setting ir sensor connected pins as input
  pinMode(irSensorBack,INPUT);
  pinMode(irSensorLeft,INPUT);
  pinMode(irSensorRight,INPUT);
  pinMode(irSensorHeadRight,INPUT);
  pinMode(irSensorHeadLeft,INPUT);

  //setting bluetoothMode & voice mode button connected pin as input as pull up mode
  pinMode(bluetoothMode,INPUT_PULLUP);
  pinMode(voiceMode, INPUT_PULLUP);

  //setting servo motor connected pins to their respective objects
  servoEyeControl.attach(servoEye);
  servoClawControl.attach(servoClaw);
  servoFrontBackControl.attach(servoFrontBack);
  servoLeftRightControl.attach(servoLeftRight);
  servoUpDownControl.attach(servoUpDown);

  Serial.begin(9600); //serial begin for debugging purpose
  // Initialise the IO and ISR
  vw_set_rx_pin(receive_pin);
  vw_set_ptt_inverted(true); // Required for DR3100
  vw_setup(2000);  // Bits per sec
  vw_rx_start(); // Start the receiver PLL running

  //initializing the robtic arm position 
  servoClawControl.write(clawMovement);
  servoFrontBackControl.write(frontBackMovement);
  servoLeftRightControl.write(leftRightMovement);
  servoUpDownControl.write(upDownMovement);
  /*
  //having problem with simulation for these part of the code
  //some fancy movement of eye servo while turning on the car for first time.
  for (int angle = 0; angle <= 180; angle++) {
    servoEyeControl.write(angle);
    delay(15); // Adjust this value for your desired speed movement
  }
  delay(100);
  for (int angle = 180; angle >= 0; angle--) {
    servoEyeControl.write(angle);
    delay(15);// Adjust this value for your desired speed movement
  }
  delay(100);
  servoEyeControl.write(forwardSeeAngle);*/
}

void loop() {
  uint8_t buf[VW_MAX_MESSAGE_LEN];
  uint8_t buflen = VW_MAX_MESSAGE_LEN;
  //checking every mode condition
  //0 means mode activated. Because we use internal pull_up resistance
  if(digitalRead(bluetoothMode)==0){
    Serial.println("Bluetooth Mode Selected---- ");
    bluetoothcontrol();
  }
  else if (digitalRead(voiceMode)==0){
    Serial.println("Voice Mode Selected---- ");
    voiceControl();
  }
  else if(vw_get_message(buf, &buflen)){
    for (int i = 0; i < buflen; i++)
    {
      char c = (buf[i]);
      message = message + c ; // make a message from the received characters         
    }
    myString = message; 
    message = "";
    String firstComma = getValue(myString, ',', 0); // xAxis
    String secondComma = getValue(myString, ',', 1); // yAxis
    String thirdComma = getValue(myString, ',', 2); // felxSensorClawValue
    String forthComma = getValue(myString, ',', 3); // clawInterrputValue
    String fifthComma = getValue(myString, ',', 4); // modeInterruptValue

    receivedData.xAxis = firstComma.toInt();
    receivedData.yAxis = secondComma.toInt();
    receivedData.felxSensorClawValue = thirdComma.toFloat();
    receivedData.clawInterrputValue = forthComma.toInt();
    receivedData.modeInterruptValue = fifthComma.toInt();

    //for debugging 
    Serial.println("Message Received through wireless Communication, Values= ");
    Serial.print("xAxis: ");
    Serial.println(receivedData.xAxis);
    Serial.print("\nyAxis: ");
    Serial.println(receivedData.yAxis);
    Serial.print("\nfelxSensorClawValue: ");
    Serial.println(receivedData.felxSensorClawValue);
    Serial.print("\Mode state: ");
    Serial.println(receivedData.modeInterruptValue);
    Serial.print("\Claw state: ");
    Serial.println(receivedData.clawInterrputValue);

    //Now remaining mode condtion will be written inside this condition
    if(receivedData.modeInterruptValue==1){
      roboticClawControl();
    }
    else if(receivedData.modeInterruptValue==2){
      gestureControl();
    }
    else if(receivedData.modeInterruptValue==3){
      lineTracking();
    }
    else if(receivedData.modeInterruptValue==4){
      obstacleAvoidance();
    }
    else if(receivedData.modeInterruptValue==5){
      obstacleFollowing();
    }
    Serial.println();
  }
  else {
    //connection failed state
    stop();
  }
}


//function for stoping the vehicle
void stop(){
  analogWrite(leftMotorEnable, 0);
  analogWrite(rightMotorEnable, 0);

  digitalWrite(leftMotorForward, LOW);
  digitalWrite(leftMotorBackward, LOW);
  digitalWrite(rightMotorForward, LOW);
  digitalWrite(rightMotorBackward, LOW);
};



//function for forward moving  
void forward(int speed){
  servoEyeControl.write(forwardSeeAngle); //this line is for tesing purpose
  long distance = readUltrasonicDistance(ultraSonicTrig, ultraSonicEcho);
  Serial.println(distance);
  if(distance<=10 && distance>0){
    stop();
    Voice="";
    voice=' ';
  }
  else{
    Serial.println("Going Forward");
    analogWrite(leftMotorEnable, speed);//setting speed
    analogWrite(rightMotorEnable, speed);

    digitalWrite(leftMotorForward, HIGH); //left motor moving forward
    digitalWrite(leftMotorBackward, LOW);
    digitalWrite(rightMotorForward, HIGH); //right motor moving forward
    digitalWrite(rightMotorBackward, LOW);
  }
}



//function for backward moving  
void backward(int speed)
{
  Serial.println("Going Backward");
  int irSensorObjectDetection = digitalRead(irSensorBack);
  if(irSensorObjectDetection == 0) { //0 means object detected
    stop();
    Voice="";
    voice=' ';
  }
  else{
    analogWrite(leftMotorEnable, speed);//setting speed
    analogWrite(rightMotorEnable, speed);

    digitalWrite(leftMotorForward, LOW);
    digitalWrite(leftMotorBackward, HIGH); //left motor moving backward
    digitalWrite(rightMotorForward, LOW); 
    digitalWrite(rightMotorBackward, HIGH);//right motor moving backward
  }
}


//function for right turning
void right(int speed){
  Serial.println("Turning right");
  analogWrite(leftMotorEnable, speed);//setting speed
  analogWrite(rightMotorEnable, speed);
  
  digitalWrite(leftMotorForward, HIGH); //left motor moving forward
  digitalWrite(leftMotorBackward, LOW);
  digitalWrite(rightMotorForward, LOW); 
  digitalWrite(rightMotorBackward, HIGH);//right motor moving backward
}


//function for left turning
void left(int speed){
  Serial.println("Turning left");
  analogWrite(leftMotorEnable, speed);//setting speed
  analogWrite(rightMotorEnable, speed);

  digitalWrite(leftMotorForward, LOW); 
  digitalWrite(leftMotorBackward, HIGH);//left motor moving backward
  digitalWrite(rightMotorForward, HIGH); //right motor moving forward
  digitalWrite(rightMotorBackward, LOW);
}

//ultrasonic sensor distance mesurement
long readUltrasonicDistance (int triggerPin, int echoPin)
{
  pinMode (triggerPin, OUTPUT); // Clear the trigger
  digitalWrite (triggerPin, LOW);
  delayMicroseconds(2);
  // Sets the trigger pin to HIGH state for 10 microseconds
  digitalWrite (triggerPin, HIGH);
  delayMicroseconds(10);
  digitalWrite (triggerPin, LOW);
  pinMode (echoPin, INPUT);
  // Reads the echo pin, and returns the sound wave travel time in microseconds
  return 0.017*pulseIn (echoPin, HIGH); //returns distance in centimeter
}

//lookLeft  function for detecting objects distance in left side
long lookLeft(){
  servoEyeControl.write(leftSeeAngle);
  delay(leftRightSeeDelay);
  long distance = readUltrasonicDistance(ultraSonicTrig, ultraSonicEcho);
  return distance;
}

//lookRight function for detecting objects distance in right side
int lookRight(){  
  servoEyeControl.write(rightSeeAngle);
  delay(leftRightSeeDelay);
  long distance = readUltrasonicDistance(ultraSonicTrig, ultraSonicEcho);
  return distance;
}

//function for line tracking mode
/*note: Line tracing mode has independent speed set parameter. 
It is not controlled by global variable. if you want to change speed in this mode,
then change the parameter value of the function here*/
void lineTracking(){
  int LEFT_SENSOR = digitalRead(irSensorLeft);
  int RIGHT_SENSOR = digitalRead(irSensorRight);
  if(RIGHT_SENSOR==0 && LEFT_SENSOR==0) {
    forward(255);
    //for debugging purpose uncomment these lines code
    /*
    Serial.print("\nLeftSensor Value = ");
    Serial.print(LEFT_SENSOR);
    Serial.print("RightSensor Value = ");
    Serial.print(RIGHT_SENSOR);
    Serial.println("Going forward");
    */
  }
  else if(RIGHT_SENSOR==0 && LEFT_SENSOR==1) {
    left(255);
    //for debugging purpose uncomment these lines code
    /*
    Serial.print("\nLeftSensor Value = ");
    Serial.print(LEFT_SENSOR);
    Serial.print("RightSensor Value = ");
    Serial.print(RIGHT_SENSOR);
    Serial.println("Going right");
    */
  }
  else if(RIGHT_SENSOR==1 && LEFT_SENSOR==0) {
    right(255);
    //for debugging purpose uncomment these lines code
    /*
    Serial.print("\nLeftSensor Value = ");
    Serial.print(LEFT_SENSOR);
    Serial.print("RightSensor Value = ");
    Serial.print(RIGHT_SENSOR);
    Serial.println("Going left");
    */
  }
  else if(RIGHT_SENSOR==1 && LEFT_SENSOR==1) {
    stop();
    //for debugging purpose uncomment these lines code
    /*
    Serial.print("\nLeftSensor Value = ");
    Serial.print(LEFT_SENSOR);
    Serial.print("RightSensor Value = ");
    Serial.print(RIGHT_SENSOR);
    Serial.println("Stopped");
    */
  }
 }


//function for bluetooth control
void bluetoothcontrol() {
  //in this condition this available function checks wheather any new data is available or not
  if(Serial.available() > 0) {
    voice = Serial.read();
    //printing command values in serial monitor
    Serial.print("Bluetooth command: ");
    Serial.println((char)voice);
  }
  //these condition will control data for bluetooth button car control
  if (voice == 'F') {
      forward(maxSpeed);
      //Serial.println("Going Forward");
  } 
  else if (voice == 'B') {
    backward(maxSpeed);
    //Serial.println("Going Backward");
  } 
  else if (voice == 'L') {
    left(maxSpeed);
    //Serial.println("Going Left");
  } 
  else if (voice == 'R') {
    right(maxSpeed);
    //Serial.println("Going Right");
  }
  else if(voice =='S'){
    stop();
    //Serial.println("Stopped");
    voice = ' ';
  }
  //car movement control ends here
  //these command will control robotic arm position by bluetooth buttons 
  else if(voice == 'f'){
    frontBackMovement+=5;
  }
  else if(voice == 'b'){
    frontBackMovement-=5;
  }
  else if(voice == 'r'){
    leftRightMovement+=5;
  }
  else if(voice == 'l'){
    leftRightMovement-=5;
  }
  else if(voice == 'u'){
    upDownMovement+=5;
  }
  else if(voice == 'd'){
    upDownMovement-=5;
  }
  else if(voice == 'o'){
    clawMovement+=5;
  }
  else if(voice == 'c'){
    clawMovement-=5;
  }
  else if(voice == 'x'){ //this condition tells to stop updating new values of robot arm
    voice = ' ';
  }
  //robotic arm position fixing code finish here
  //now this if condition will check if any robotic arm command came or not
  if(voice=='f' || voice=='b' ||voice=='r' ||voice=='l' ||voice=='u' ||voice=='d' ||voice=='o' ||voice=='c'){
    roboticArmPositionUpdate();//updating robotic arm position
    //Serial.println("Updating robotic arm position");
  }
}

//function for controlling car using voice command via phone
void voiceControl(){
  //Now thsese condition will control the car by voice 
  if(Serial.available() > 0){
    Voice = Serial.readString();
    //printing command values in serial monitor
    Serial.print("Voice command: ");
    Serial.println(Voice);
  }
  if (Voice == "forward") {
   forward(maxSpeed);
   //Serial.println("Going Forward");
  } 
  else if (Voice == "backward") {
    backward(maxSpeed);
    //Serial.println("Going Backward");
  } 
  else if (Voice == "left") {
    long leftDistance = lookLeft();
    //if no obstacle in left then turning left
    if(leftDistance>10){
      left(maxSpeed);
      //Serial.println("Turning left");
      delay(leftRightTurningDelay);
      stop();
    }
    else{
      //some fancy movement like saying no in sign language
      servoEyeControl.write(forwardSeeAngle);
      delay(150);
      servoEyeControl.write(120);
      delay(150);
      servoEyeControl.write(60);
      delay(150);
    }
    servoEyeControl.write(forwardSeeAngle); //repositioning eye
  } 
  else if (Voice == "right") {
    //watching right side 
    long rightDistance = lookRight();
    //if no obstacle in left then turning left
    if(rightDistance>10){
      right(maxSpeed);
      //Serial.println("Turning right");
      delay(leftRightTurningDelay);
      stop();
    }
    else{
      //some fancy movement like saying no in sign language
      servoEyeControl.write(forwardSeeAngle);
      delay(150);
      servoEyeControl.write(120);
      delay(150);
      servoEyeControl.write(60);
      delay(150);
    }
    servoEyeControl.write(forwardSeeAngle); //repositioning eye
  }
  else if (Voice == "stop") {
    stop();
    //Serial.println("Stopped");
  }
  Voice = "";  
  //voice control command ends here
}



//function for automatic obstacle avoiding mode
void obstacleAvoidance(){
  servoEyeControl.write(forwardSeeAngle);
  long distance = readUltrasonicDistance(ultraSonicTrig, ultraSonicEcho);
  //printing front distance in serial monitor for debugging
  
  Serial.print("Distance: ");
  Serial.println(distance);
  
  /*if there is an obstacle in front of our car (approximetly 20cm away), 
    then it will stop and decide wheather to go right or left*/
  if(distance< 20){
    stop(); //stop for 0.5 seconds
    delay(500);
    backward(maxSpeed); // go backward at maxSpeed pwm value speed for 0.2 seconds
    delay(200);
    stop(); // after 0.2 second backward, stop here.

    long leftDistance = lookLeft(); //store left size obstacle distance
    delay(400);
    long rightDistance = lookRight(); //store right side obstacle distance 
    delay(400);
    servoEyeControl.write(forwardSeeAngle);

    //printing left and right distance for debugging
    
    Serial.print("Left Distance: ");
    Serial.println(leftDistance);
    Serial.print("Right Distance: ");
    Serial.println(rightDistance);
    Serial.print("\n\n");
    

    if(leftDistance>rightDistance){
      left(maxSpeed);
      delay(leftRightTurningDelay);
      stop();
    }
    else if(leftDistance<rightDistance){
      right(maxSpeed);
      delay(leftRightTurningDelay);
      stop();
    }
    else{ //if both left and right distance are equal then randomly change direction
      int randomTurn = random(2); // Generate a random number: 0 or 1
      if (randomTurn == 0) {
        right(maxSpeed);
        delay(leftRightTurningDelay);
        stop();
      } 
      else{
        left(maxSpeed);
        delay(leftRightTurningDelay);
        stop();
      }
    }
  }
  //if no obstacle in front then simply go forward
  else{
    Serial.println("Obstacle not detected");
    forward(maxSpeed);
  }
}



//function for gesture control mode
void gestureControl(){
  if(receivedData.yAxis>minSpeed){
    forward(receivedData.yAxis);
    /*
    Serial.print("Forward mode value: ");
    Serial.println(receivedData.yAxis);
    */
  }
  else if(receivedData.yAxis<-minSpeed){
    backward(abs(receivedData.yAxis));
    /*
    Serial.print("Backward mode value: ");
    Serial.println(receivedData.yAxis);
    */
  }
  else if(receivedData.xAxis>minSpeed){
    right(receivedData.xAxis);
    /*
    Serial.print("Right mode value: ");
    Serial.println(receivedData.xAxis);
    */
  }
  else if(receivedData.xAxis<-minSpeed){
    left(abs(receivedData.xAxis));
    /*
    Serial.print("Left mode value: ");
    Serial.println(receivedData.yAxis);
    */
  }
  else{
    stop();
    /*
    Serial.println("Stop mode value: ");
    Serial.print("X axis value: ");
    Serial.println(receivedData.xAxis);
    Serial.print("Y axis value: ");
    Serial.println(receivedData.yAxis);
    */
  }
}


//function for robotic arm controling mode
void roboticClawControl(){
  //checking up down movement was selected by the push button or not
  static unsigned long time1 = millis();
  unsigned long time2 = millis();
  unsigned long difference = time2-time1;
  Serial.print("Time elapsed between Arm movement: ");
  Serial.println(difference);
  if (difference >150){
    if(receivedData.clawInterrputValue==0){
    //checking front and back movement
      if(receivedData.yAxis>100){
        if(frontBackMovement<180) {frontBackMovement += armStepSize; positionChange = true;}
      }
      else if(receivedData.yAxis<-100){
        if(frontBackMovement>0) {frontBackMovement -= armStepSize; positionChange = true;}
      }

      //checking Left and Right movement
      if(receivedData.xAxis>100){
        if(leftRightMovement<180) {leftRightMovement += armStepSize; positionChange = true;}
      }
      else if(receivedData.xAxis<-100){
        if(leftRightMovement>0) {leftRightMovement -= armStepSize; positionChange = true;}
      }

      //checking claw open and close movement
      if(receivedData.felxSensorClawValue<35 && receivedData.felxSensorClawValue>15){ //means flex sensor is almost fully open
        if(clawMovement<180) {clawMovement+=armStepSize; positionChange = true;}
      }
      else if(receivedData.felxSensorClawValue>60){ //means flex sensor is bending more than 60 degrees 
        if(clawMovement>0) {clawMovement-=armStepSize; positionChange = true;}
      }
   }
    //else condition means that push button is pressed and up down movement selected
    else{
      //checking up and down movement
      if(receivedData.felxSensorClawValue<35 && receivedData.felxSensorClawValue>15){ //means flex sensor is almost fully open
          if(upDownMovement<180) {upDownMovement += armStepSize; positionChange = true;}
      }
      else if(receivedData.felxSensorClawValue>60){ //means flex sensor is bending more than 60 degrees 
        if(upDownMovement>0) {upDownMovement -= armStepSize; positionChange = true;}
      }
    }

    if(positionChange == true){
      //for debugging purpose
      Serial.println("Robot hand updated values---");
      Serial.print("frontBackMovement= ");
      Serial.println(frontBackMovement);
      Serial.print("leftRightMovement= ");
      Serial.println(leftRightMovement);
      Serial.print("upDownMovement= ");
      Serial.println(upDownMovement);
      Serial.print("clawMovement= ");
      Serial.println(clawMovement);
      roboticArmPositionUpdate();//calling this function for updating position of robotic arm
      positionChange = false;
    }
    time1 = time2;
  }  
}

//this is a function for moving robotic arm 
void roboticArmPositionUpdate(){
  /*setting the maximum and minimum boundaries of our robotic arm servo motor angle
    change those min and max value accordingly to your claw for best result*/
    /*
  frontBackMovement = min(145, max(15, frontBackMovement)); //here 15 degree is minimum value and 145 degree is maximum value
  leftRightMovement = min(175, max(40, leftRightMovement));
  upDownMovement = min(150, max(5, upDownMovement));
  clawMovement = min(90, max(75, clawMovement));
  */

  Serial.println("Arm Position updated");
  //now update the servo motor position for robotic arm movement
  servoFrontBackControl.write(frontBackMovement);
  servoLeftRightControl.write(leftRightMovement);
  servoUpDownControl.write(upDownMovement);
  servoClawControl.write(clawMovement);
}

//function for obstacle following mode
void obstacleFollowing(){
  long distance = readUltrasonicDistance(ultraSonicTrig, ultraSonicEcho); //read distance value
  unsigned int Right_Value = digitalRead(irSensorHeadRight);             // read the value from Right IR sensor:
  unsigned int Left_Value = digitalRead(irSensorHeadLeft);               // read the value from Left IR sensor:
 
  if((distance > 1) && (distance < 15)){ //check wheather the ultrasonic sensor's value stays between 1 to 15.
  /*if it is true then it will go forward. But we can't use forward function 
    cause its designed to avoid obstacle instead of following.
    That's why re-writting the function with a suitable speed value*/

    analogWrite(leftMotorEnable, 255);//setting speed
    analogWrite(rightMotorEnable, 255);

    digitalWrite(leftMotorForward, HIGH); //left motor moving forward
    digitalWrite(leftMotorBackward, LOW);
    digitalWrite(rightMotorForward, HIGH); //right motor moving forward
    digitalWrite(rightMotorBackward, LOW);
  }
  else if((Right_Value==0) && (Left_Value==1)){
    left(255);
  }
  else if((Right_Value==1) && (Left_Value==0)){
    right(255);
  }
  else if(distance > 15) {
    stop();
  }
}

//this function is for getting values from the string separated by commas
String getValue(String data, char separator, int index)
{
    int found = 0;
    int strIndex[] = { 0, -1 };
    int maxIndex = data.length() - 1;

    for (int i = 0; i <= maxIndex && found <= index; i++) {
        if (data.charAt(i) == separator || i == maxIndex) {
            found++;
            strIndex[0] = strIndex[1] + 1;
            strIndex[1] = (i == maxIndex) ? i+1 : i;
        }
    }
    return found > index ? data.substring(strIndex[0], strIndex[1]) : "";
}