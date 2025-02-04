#include <Wire.h>
#include <MPU6050.h>
#include <nRF24L01.h>
#include <RF24.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

//defining all pin number in which the component is onnected
#define I2C_ADDRESS 0x3C // Address depends on I2C module. need to be changed for every different module 

#define CE 7  //pin no for nrf24l01 CE pin
#define CSN 8 //pin no for nrf24l01 CSN pin

#define flexSensorClaw A6 //pin no for flex sensor which control robot claw open and close operation
#define clawInterrput 2 //push button for claw movement change in up down position
#define modeInterrupt 3 //push button for mode change 

volatile int mode = 0;
volatile bool clawState = 0;
const float flatResistance= 25000.0; //change this value according to sensor value when it is in straight position
const float bendResistance = 75000.0; //change this value according to sensor value when it is in 90 degree bend position
const float pullDownResistance = 47000.0; //change this value according the resistor value you used in pull down purpose
const int delayAfterSendingData = 10; //this value determines how frequently the data will transfer.
const int maxErrorCount = 5; // this variable defines maximum failed attempt of data transmission
int errorCount = 0; //this variable is for current failed attemp for data transmission 

//creating a structure to store all the data in one package
struct data {
  int xAxis;
  int yAxis;
  float felxSensorClawValue;
  bool clawInterrputValue;
  int modeInterruptValue;
};

//creating necessary objects 
Adafruit_SSD1306 lcd = Adafruit_SSD1306(128, 32, &Wire);
MPU6050 mpu;//This is for gyro sensor
RF24 transmitter(CE, CSN); //this is for nrf24 module to transmit wireless data
const byte address[6] = "1Node"; //5 digit address. This address should be same in receiver side also
data sendData; // creating a data object to store data in that object

void setup() {
  //setting pinmodes
  pinMode(flexSensorClaw, INPUT);
  /*
  using internal pull up resistor. so we need to connect button with ground
  For that, when the button pressed the arduino will read LOW as its button state
  and it will remain HIGH sate if button is at off state
  */
  pinMode(clawInterrput, INPUT_PULLUP);
  pinMode(modeInterrupt, INPUT_PULLUP);

  //attaching interrupt to change value in Rising state of interrupt pin
  attachInterrupt(digitalPinToInterrupt(clawInterrput),clawStateChange, RISING);
  attachInterrupt(digitalPinToInterrupt(modeInterrupt),modeChange, RISING);

  Wire.begin();
  Serial.begin(9600);
  mpu.initialize();
  transmitter.begin();
  transmitter.openWritingPipe(address);  //setting address
  //transmitter.setPALevel(RF24_PA_HIGH); //setting power consumption of nrf module max for maximum range
  transmitter.setPayloadSize(sizeof(data)); // setting size of data need to be transfered 
  transmitter.setDataRate(RF24_250KBPS); //setting data rate low of nrf module for max range
  transmitter.stopListening(); // as it doesn't revice any data we need to stop receiving data

  lcd.begin(SSD1306_SWITCHCAPVCC, I2C_ADDRESS); // Address for display
  lcd.clearDisplay();     // Clear the display
  lcd.setTextColor(WHITE); //setting text color
  lcd.setTextSize(2); //text size defining
  lcd.setCursor(15, 5); //set the text position to start from
  lcd.print("Welcome!");  //writting a welcome sms for 2 sec when it is turning on. 
  lcd.display();
  delay(2000);
}

void loop() {

    int accelerometerX, accelerometerY, accelerometerZ;
    mpu.getAcceleration(&accelerometerY, &accelerometerX, &accelerometerZ); // getting value from gyro sensor mpu6050 

    sendData.xAxis = map(accelerometerX, -16384, 16384, -255, 255);//mapping value in this range for pwm motor speed control
    sendData.yAxis = map(accelerometerY, -16384, 16384, -255, 255);
    //Ensuring the values are within bounds
    sendData.xAxis = constrain(sendData.xAxis, -255, 255);
    sendData.yAxis = constrain(sendData.yAxis, -255, 255)*(-1);
    
    sendData.felxSensorClawValue = flexAngelCalculator(flexSensorClaw); //sending the flexing angle of the sensor
    
    sendData.clawInterrputValue = clawState;
    sendData.modeInterruptValue = mode;

    //uncomment below code if you want to print values for debugging purpose
    
    Serial.print("xAxis: ");
    Serial.print(sendData.xAxis);
    Serial.print("\nyAxis: ");
    Serial.println(sendData.yAxis);
    Serial.print("\nfelxSensorClawValue: ");
    Serial.println(sendData.felxSensorClawValue);
    Serial.print("\Mode state: ");
    Serial.println(sendData.modeInterruptValue);
    Serial.print("\Claw state: ");
    Serial.println(sendData.clawInterrputValue);
    
    //sending the data and storing its status in a variable
    bool status = transmitter.write(&sendData, sizeof(data)); 

    /* setting a static variable "displayMessage" cause it will run only one time
    for this variable the display won't refresh for same text over and over again*/
    static String displayMessage = "Faysal-Shairy-Antik-Ajmy-Prottoy Made this code!"; 

    if(sendData.modeInterruptValue==0) { //no mode selected
      if(displayMessage != "Select a mode!"){
        lcd.clearDisplay();  //clear the display first
        lcd.setTextSize(1); //text size defining
        lcd.setCursor(20, 10); //set the text position to start from
        lcd.print("Select a mode!");  //write the text you want to display
        lcd.display();
        displayMessage = "Select a mode!"; //updating data
      }
    }
    else if(status == 0){
      errorCount++; //increasing error count for failed status
      //if maximum number of attempt was failed to send data then display error sms
      if(errorCount>=maxErrorCount){
        if(displayMessage!="Error! Data sending failed"){
          lcd.clearDisplay(); // clear the display first
          lcd.setTextSize(2); //text size defining
          lcd.setCursor(28, 0); //set the text position to start from
          lcd.print("Error!");  //write the text you want to display
          lcd.setTextSize(1); //text size defining
          lcd.setCursor(5, 20); //set the text position to start from
          lcd.print("Data sending failed");  //write the text you want to display
          lcd.display();
          displayMessage = "Error! Data sending failed"; //updating data
        }
      }
    }
    else if(sendData.modeInterruptValue==1){
      errorCount=0; //clearing unsuccessfull attempt
      if(displayMessage!= "Claw Control Mode"){
        lcd.clearDisplay(); // clear the display first
        lcd.setTextSize(2); //text size defining
        lcd.setCursor(35, 0); //set the text position to start from
        lcd.print("Claw");  //write the text you want to display
        lcd.setTextSize(1); //text size defining
        lcd.setCursor(25, 20); //set the text position to start from
        lcd.print("Control Mode");  //write the text you want to display
        lcd.display();
        displayMessage = "Claw Control Mode"; //updating data
      }
    }
    else if(sendData.modeInterruptValue==2){
      errorCount=0; //clearing unsuccessfull attempt
      if(displayMessage!= "Gesture Control Mode"){
        lcd.clearDisplay(); // clear the display first
        lcd.setTextSize(2); //text size defining
        lcd.setCursor(18, 0); //set the text position to start from
        lcd.print("Gesture");  //write the text you want to display
        lcd.setTextSize(1); //text size defining
        lcd.setCursor(25, 20); //set the text position to start from
        lcd.print("Control Mode");  //write the text you want to display
        lcd.display();
        displayMessage = "Gesture Control Mode"; //updating data
      }
    }
    else if(sendData.modeInterruptValue==3){
      errorCount=0; //clearing unsuccessfull attempt
      if(displayMessage!= "Line Following Mode"){
        lcd.clearDisplay(); // clear the display first
        lcd.setTextSize(2); //text size defining
        lcd.setCursor(38, 0); //set the text position to start from
        lcd.print("Line");  //write the text you want to display
        lcd.setTextSize(1); //text size defining
        lcd.setCursor(22, 20); //set the text position to start from
        lcd.print("Following Mode");  //write the text you want to display
        lcd.display();
        displayMessage = "Line Following Mode"; //updating data
      }
    }
    else if(sendData.modeInterruptValue==4){
      errorCount=0; //clearing unsuccessfull attempt
      if(displayMessage!= "Obastacle Avoidance Mode"){
        lcd.clearDisplay(); // clear the display first
        lcd.setTextSize(2); //text size defining
        lcd.setCursor(15, 0); //set the text position to start from
        lcd.print("Obstacle");  //write the text you want to display
        lcd.setTextSize(1); //text size defining
        lcd.setCursor(22, 20); //set the text position to start from
        lcd.print("Avoidance Mode");  //write the text you want to display
        lcd.display();
        displayMessage = "Obastacle Avoidance Mode"; //updating data
      }
    }
    else if(sendData.modeInterruptValue==5){
      errorCount=0; //clearing unsuccessfull attempt
      if(displayMessage!= "Obastacle Following Mode"){
        lcd.clearDisplay(); // clear the display first
        lcd.setTextSize(2); //text size defining
        lcd.setCursor(15, 0); //set the text position to start from
        lcd.print("Obstacle");  //write the text you want to display
        lcd.setTextSize(1); //text size defining
        lcd.setCursor(22, 20); //set the text position to start from
        lcd.print("Following Mode");  //write the text you want to display
        lcd.display();
        displayMessage = "Obastacle Following Mode"; //updating data
      }
    }
    delay(delayAfterSendingData); //loops runs after halting for desired time
}

float flexAngelCalculator(const unsigned char sensorConnectedPin){
  int ADCflexPDV = analogRead(sensorConnectedPin);
	float felxPDV = ADCflexPDV * 5 / 1023.0;
	float flexSensorResistance = pullDownResistance * (5 / felxPDV - 1.0);
	// Use the calculated resistance to estimate the sensor's bend angle:
	float angle = map(flexSensorResistance, flatResistance, bendResistance, 0, 90.0);

  return angle;
}

//mode change interrupt function
void modeChange(){
  if(mode<5){
    mode++;
  }
  else{
    mode = 0;
  }
}

//claw state mode changing interrupt function 
void clawStateChange(){
  clawState=!clawState;
}