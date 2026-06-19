/*
 * Smart Helmet Bike Ignition System — ESP32 Firmware
 * Author: Final Year Project Team
 *
 * Receives ignition commands from Raspberry Pi via:
 *   - Serial UART (USB cable)
 *   - WiFi/MQTT (wireless)
 *
 * Controls:
 *   - Relay module → Bike ignition relay
 *   - LED indicators (Green = ON, Red = OFF)
 *   - LCD display (I2C 16x2)
 *   - Buzzer (alert tones)
 *
 * Pin Map (ESP32 DevKit V1):
 *   GPIO 26 → Relay IN (Active LOW)
 *   GPIO 25 → Green LED (Ignition Enabled)
 *   GPIO 33 → Red LED   (Ignition Locked)
 *   GPIO 32 → Buzzer
 *   GPIO 21 → SDA (LCD I2C)
 *   GPIO 22 → SCL (LCD I2C)
 *   GPIO 1  → TX (Serial to Pi)
 *   GPIO 3  → RX (Serial from Pi)
 */

// ─────────────────────────────────────────────
// LIBRARIES
// ─────────────────────────────────────────────
#include <Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// WiFi + MQTT (optional — comment out if using serial only)
#include <WiFi.h>
#include <PubSubClient.h>

// ─────────────────────────────────────────────
// CONFIGURATION — EDIT THESE
// ─────────────────────────────────────────────
#define USE_WIFI       true     // Set false for serial-only mode
#define USE_SERIAL     true     // Always keep true for debugging

const char* WIFI_SSID     = "YourWiFiSSID";
const char* WIFI_PASSWORD = "YourWiFiPassword";
const char* MQTT_SERVER   = "192.168.1.100";
const int   MQTT_PORT     = 1883;
const char* MQTT_TOPIC_SUB  = "helmet/ignition";
const char* MQTT_TOPIC_PUB  = "helmet/esp32/status";
const char* MQTT_CLIENT_ID  = "ESP32_HelmetSystem";

// ─────────────────────────────────────────────
// HARDWARE PINS
// ─────────────────────────────────────────────
#define PIN_RELAY       26   // Active LOW relay
#define PIN_LED_GREEN   25
#define PIN_LED_RED     33
#define PIN_BUZZER      32
#define PIN_LCD_SDA     21
#define PIN_LCD_SCL     22

// ─────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────
#define RELAY_ON        LOW    // Most relay modules are active LOW
#define RELAY_OFF       HIGH
#define SERIAL_BAUD     115200
#define WATCHDOG_MS     30000  // Lock ignition if no signal for 30 sec
#define BUZZER_FREQ_ON  1000
#define BUZZER_FREQ_OFF 400

// ─────────────────────────────────────────────
// OBJECTS
// ─────────────────────────────────────────────
LiquidCrystal_I2C lcd(0x27, 16, 2);   // I2C address 0x27 (check with scanner)

WiFiClient   wifiClient;
PubSubClient mqttClient(wifiClient);

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────
bool  ignitionEnabled  = false;
bool  helmetDetected   = false;
unsigned long lastSignalMs = 0;
unsigned long systemStartMs = 0;

String serialBuffer = "";

// ─────────────────────────────────────────────
// FUNCTION PROTOTYPES
// ─────────────────────────────────────────────
void setupPins();
void setupLCD();
void setupWiFi();
void setupMQTT();
void handleSerial();
void handleMQTT();
void setIgnition(bool enable, const char* source);
void updateLCD(const char* line1, const char* line2);
void beep(int freq, int durationMs, int times = 1);
void checkWatchdog();
void publishStatus();
void mqttCallback(char* topic, byte* payload, unsigned int length);
void reconnectMQTT();

// ─────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────
void setup() {
  Serial.begin(SERIAL_BAUD);
  systemStartMs = millis();

  Serial.println("\n========================================");
  Serial.println("  Smart Helmet Ignition System");
  Serial.println("  ESP32 Firmware v1.0");
  Serial.println("========================================");

  setupPins();
  setupLCD();

  updateLCD("Helmet System", "Initializing...");
  delay(1000);

  if (USE_WIFI) {
    setupWiFi();
    setupMQTT();
  }

  // Start in locked state
  setIgnition(false, "BOOT");
  updateLCD("Wear Helmet to", "Start Bike");
  Serial.println("[READY] System ready. Waiting for detection signal.");
}

// ─────────────────────────────────────────────
// MAIN LOOP
// ─────────────────────────────────────────────
void loop() {
  if (USE_SERIAL) {
    handleSerial();
  }

  if (USE_WIFI && mqttClient.connected()) {
    mqttClient.loop();
  } else if (USE_WIFI) {
    reconnectMQTT();
  }

  checkWatchdog();
  delay(10);
}

// ─────────────────────────────────────────────
// PIN SETUP
// ─────────────────────────────────────────────
void setupPins() {
  pinMode(PIN_RELAY,     OUTPUT);
  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_RED,   OUTPUT);
  pinMode(PIN_BUZZER,    OUTPUT);

  // Safe defaults
  digitalWrite(PIN_RELAY,     RELAY_OFF);
  digitalWrite(PIN_LED_GREEN, LOW);
  digitalWrite(PIN_LED_RED,   HIGH);   // Red ON = locked
  digitalWrite(PIN_BUZZER,    LOW);
}

// ─────────────────────────────────────────────
// LCD SETUP
// ─────────────────────────────────────────────
void setupLCD() {
  Wire.begin(PIN_LCD_SDA, PIN_LCD_SCL);
  lcd.init();
  lcd.backlight();
  lcd.clear();
}

// ─────────────────────────────────────────────
// WIFI SETUP
// ─────────────────────────────────────────────
void setupWiFi() {
  Serial.print("[WiFi] Connecting to: ");
  Serial.println(WIFI_SSID);
  updateLCD("WiFi Connecting", WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected!");
    Serial.print("[WiFi] IP: ");
    Serial.println(WiFi.localIP());
    updateLCD("WiFi Connected", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[WiFi] Failed. Serial-only mode.");
    updateLCD("WiFi Failed", "Serial Mode");
  }
  delay(1000);
}

// ─────────────────────────────────────────────
// MQTT SETUP
// ─────────────────────────────────────────────
void setupMQTT() {
  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
  reconnectMQTT();
}

void reconnectMQTT() {
  if (mqttClient.connected()) return;
  if (millis() - lastSignalMs < 5000) return;  // Retry delay

  Serial.print("[MQTT] Connecting...");
  if (mqttClient.connect(MQTT_CLIENT_ID)) {
    Serial.println(" connected!");
    mqttClient.subscribe(MQTT_TOPIC_SUB);
    Serial.print("[MQTT] Subscribed: ");
    Serial.println(MQTT_TOPIC_SUB);
  } else {
    Serial.print(" failed, rc=");
    Serial.println(mqttClient.state());
  }
}

// ─────────────────────────────────────────────
// MQTT CALLBACK
// ─────────────────────────────────────────────
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  Serial.print("[MQTT] Received on ");
  Serial.print(topic);
  Serial.print(": ");
  Serial.println(message);

  lastSignalMs = millis();

  // Parse simple JSON: {"ignition": true/false, ...}
  if (message.indexOf("\"ignition\": true") >= 0 ||
      message.indexOf("\"ignition\":true") >= 0) {
    setIgnition(true, "MQTT");
  } else if (message.indexOf("\"ignition\": false") >= 0 ||
             message.indexOf("\"ignition\":false") >= 0) {
    setIgnition(false, "MQTT");
  }
}

// ─────────────────────────────────────────────
// SERIAL HANDLER
// ─────────────────────────────────────────────
void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        processCommand(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  Serial.print("[CMD] Received: ");
  Serial.println(cmd);

  lastSignalMs = millis();

  if (cmd == "IGNITION:ON") {
    setIgnition(true, "SERIAL");
  } else if (cmd == "IGNITION:OFF") {
    setIgnition(false, "SERIAL");
  } else if (cmd == "STATUS") {
    publishStatus();
  } else if (cmd == "PING") {
    Serial.println("PONG");
  } else {
    Serial.print("[CMD] Unknown command: ");
    Serial.println(cmd);
  }
}

// ─────────────────────────────────────────────
// IGNITION CONTROL — CORE FUNCTION
// ─────────────────────────────────────────────
void setIgnition(bool enable, const char* source) {
  if (enable == ignitionEnabled) return;  // No change

  ignitionEnabled = enable;
  Serial.print("[IGNITION] ");
  Serial.print(enable ? "ENABLED" : "DISABLED");
  Serial.print(" (source: ");
  Serial.print(source);
  Serial.println(")");

  if (enable) {
    // ── ENABLE IGNITION ──
    digitalWrite(PIN_RELAY,     RELAY_ON);
    digitalWrite(PIN_LED_GREEN, HIGH);
    digitalWrite(PIN_LED_RED,   LOW);
    beep(BUZZER_FREQ_ON, 150, 2);
    updateLCD("Helmet: DETECTED", "Ignition: ON  ");
    helmetDetected = true;
  } else {
    // ── DISABLE IGNITION ──
    digitalWrite(PIN_RELAY,     RELAY_OFF);
    digitalWrite(PIN_LED_GREEN, LOW);
    digitalWrite(PIN_LED_RED,   HIGH);
    beep(BUZZER_FREQ_OFF, 300, 1);
    updateLCD("Wear Helmet to", "Start Bike");
    helmetDetected = false;
  }

  publishStatus();
}

// ─────────────────────────────────────────────
// WATCHDOG — Lock if no signal received
// ─────────────────────────────────────────────
void checkWatchdog() {
  if (!ignitionEnabled) return;
  if (millis() - lastSignalMs > WATCHDOG_MS) {
    Serial.println("[WATCHDOG] No signal for 30s. Locking ignition.");
    setIgnition(false, "WATCHDOG");
    updateLCD("Signal Lost!", "Ignition Locked");
  }
}

// ─────────────────────────────────────────────
// LCD UPDATE
// ─────────────────────────────────────────────
void updateLCD(const char* line1, const char* line2) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(line1);
  lcd.setCursor(0, 1);
  lcd.print(line2);
}

// ─────────────────────────────────────────────
// BUZZER
// ─────────────────────────────────────────────
void beep(int freq, int durationMs, int times) {
  for (int i = 0; i < times; i++) {
    tone(PIN_BUZZER, freq, durationMs);
    delay(durationMs + 50);
  }
  noTone(PIN_BUZZER);
}

// ─────────────────────────────────────────────
// PUBLISH STATUS
// ─────────────────────────────────────────────
void publishStatus() {
  unsigned long uptime = (millis() - systemStartMs) / 1000;
  char payload[200];
  snprintf(payload, sizeof(payload),
    "{\"ignition\":%s,\"helmet\":%s,\"uptime_s\":%lu,\"source\":\"ESP32\"}",
    ignitionEnabled ? "true" : "false",
    helmetDetected  ? "true" : "false",
    uptime
  );

  Serial.print("[STATUS] ");
  Serial.println(payload);

  if (mqttClient.connected()) {
    mqttClient.publish(MQTT_TOPIC_PUB, payload);
  }
}
