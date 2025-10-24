#include "pyCommsLib.h"
#include <Servo.h>

Servo myServo;        // Create a servo object
int servoPin = 6;     // Servo signal pin

void setup() {
  Serial.begin(115200);
 
  init_python_communication();

  myServo.attach(servoPin);
  
  // For debugging
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

}

void loop() {
  String msg = latest_received_msg();

  if (msg.startsWith("pwm")) {
    int angle = 90; // default
    int sep = msg.indexOf(':');
    if (sep != -1) {
      angle = msg.substring(sep + 1).toInt();
    }

    angle = constrain(angle, 0, 180);
    myServo.write(angle);

    // Debug
    digitalWrite(LED_BUILTIN, HIGH); // Make sure message received
    Serial.print("Received: ");
    Serial.print(msg);
    Serial.print(" | Servo angle: ");
    Serial.println(angle);

  } 
  else { 
    digitalWrite(LED_BUILTIN, LOW);
  }

  sync();
  delay(1000);
}

