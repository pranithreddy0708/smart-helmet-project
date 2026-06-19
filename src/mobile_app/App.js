/**
 * Smart Helmet Bike Ignition System — Mobile Dashboard
 * React Native App (Expo)
 *
 * Features:
 *  - Real-time helmet detection status via MQTT/WebSocket
 *  - Ignition status indicator
 *  - Historical detection log
 *  - Notifications when helmet removed while ignition on
 *
 * Setup:
 *  npm install -g expo-cli
 *  expo install expo-notifications mqtt
 *  expo start
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TouchableOpacity,
  Alert, Vibration, Platform, StatusBar
} from 'react-native';
import * as Notifications from 'expo-notifications';

// ─────────────────────────────────────────────
// CONFIGURATION
// ─────────────────────────────────────────────
const CONFIG = {
  mqttBroker:   'ws://192.168.1.100:9001',  // WebSocket MQTT (port 9001)
  topicStatus:  'helmet/status',
  topicIgnition:'helmet/ignition',
  topicESP32:   'helmet/esp32/status',
  refreshRate:  1000,   // UI refresh ms
  maxLogItems:  50,
};

// ─────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────
const STATUS = {
  HELMET_ON_IGNITION_ON:  { label: 'HELMET ON · IGNITION ENABLED',  color: '#00C853', icon: '✅' },
  HELMET_OFF_IGNITION_OFF:{ label: 'NO HELMET · IGNITION LOCKED',   color: '#D50000', icon: '🔒' },
  HELMET_ON_IGNITION_OFF: { label: 'HELMET ON · STARTING...',       color: '#FF6D00', icon: '⏳' },
  CONNECTING:             { label: 'CONNECTING...',                  color: '#607D8B', icon: '📡' },
  DISCONNECTED:           { label: 'DISCONNECTED',                   color: '#795548', icon: '❌' },
};

// ─────────────────────────────────────────────
// MAIN APP COMPONENT
// ─────────────────────────────────────────────
export default function App() {
  const [helmetDetected, setHelmetDetected]     = useState(false);
  const [ignitionEnabled, setIgnitionEnabled]   = useState(false);
  const [confidence, setConfidence]             = useState(0);
  const [connected, setConnected]               = useState(false);
  const [lastUpdate, setLastUpdate]             = useState(null);
  const [log, setLog]                           = useState([]);
  const [esp32Status, setEsp32Status]           = useState(null);
  const [detectionCount, setDetectionCount]     = useState(0);

  const mqttRef = useRef(null);

  // ── MQTT Connection ──────────────────────────
  useEffect(() => {
    connectMQTT();
    setupNotifications();
    return () => {
      if (mqttRef.current) mqttRef.current.end();
    };
  }, []);

  const connectMQTT = () => {
    try {
      // Using mqtt.js for React Native
      // Install: npm install mqtt
      const mqtt = require('mqtt');
      const client = mqtt.connect(CONFIG.mqttBroker, {
        clientId: `mobile_dashboard_${Math.random().toString(16).substr(2, 8)}`,
        reconnectPeriod: 3000,
      });

      client.on('connect', () => {
        console.log('[MQTT] Connected');
        setConnected(true);
        client.subscribe([CONFIG.topicStatus, CONFIG.topicESP32]);
        addLog('MQTT connected', 'system');
      });

      client.on('message', (topic, payload) => {
        try {
          const data = JSON.parse(payload.toString());
          handleMessage(topic, data);
        } catch (e) {
          console.warn('[MQTT] Parse error:', e);
        }
      });

      client.on('disconnect', () => {
        setConnected(false);
        addLog('MQTT disconnected', 'error');
      });

      client.on('error', (err) => {
        console.error('[MQTT] Error:', err);
        setConnected(false);
      });

      mqttRef.current = client;
    } catch (err) {
      console.error('MQTT setup error:', err);
      // Demo mode: simulate messages
      startDemoMode();
    }
  };

  // ── DEMO MODE (no MQTT broker) ───────────────
  const startDemoMode = () => {
    addLog('Running in DEMO mode', 'system');
    let toggle = false;
    const interval = setInterval(() => {
      toggle = !toggle;
      const data = {
        helmet_detected: toggle,
        ignition_enabled: toggle,
        confidence: toggle ? 0.92 : 0.0,
        timestamp: new Date().toISOString(),
      };
      handleMessage(CONFIG.topicStatus, data);
    }, 3000);
    return () => clearInterval(interval);
  };

  // ── Message Handler ──────────────────────────
  const handleMessage = (topic, data) => {
    const now = new Date();
    setLastUpdate(now.toLocaleTimeString());

    if (topic === CONFIG.topicStatus) {
      const prevHelmet  = helmetDetected;
      const prevIgnition = ignitionEnabled;

      setHelmetDetected(data.helmet_detected);
      setIgnitionEnabled(data.ignition_enabled);
      setConfidence(data.confidence || 0);
      setDetectionCount(c => c + 1);

      const status = data.helmet_detected ? '✅ Helmet ON' : '❌ No Helmet';
      const ignition = data.ignition_enabled ? '🔑 Ignition ON' : '🔒 Locked';
      addLog(`${status} | ${ignition}`, data.helmet_detected ? 'success' : 'warning');

      // Notify if helmet removed while ignition was on
      if (prevIgnition && !data.ignition_enabled && !data.helmet_detected) {
        sendNotification('⚠️ Helmet Removed', 'Bike ignition has been locked!');
        Vibration.vibrate([0, 300, 100, 300]);
      }
    } else if (topic === CONFIG.topicESP32) {
      setEsp32Status(data);
    }
  };

  // ── Notifications ────────────────────────────
  const setupNotifications = async () => {
    const { status } = await Notifications.requestPermissionsAsync();
    if (status !== 'granted') return;
  };

  const sendNotification = async (title, body) => {
    await Notifications.scheduleNotificationAsync({
      content: { title, body, sound: true },
      trigger: null,
    });
  };

  // ── Log Helper ───────────────────────────────
  const addLog = (message, type = 'info') => {
    const entry = {
      id: Date.now(),
      time: new Date().toLocaleTimeString(),
      message,
      type,
    };
    setLog(prev => [entry, ...prev].slice(0, CONFIG.maxLogItems));
  };

  // ── Current Status ───────────────────────────
  const getCurrentStatus = () => {
    if (!connected) return STATUS.DISCONNECTED;
    if (helmetDetected && ignitionEnabled) return STATUS.HELMET_ON_IGNITION_ON;
    if (helmetDetected && !ignitionEnabled) return STATUS.HELMET_ON_IGNITION_OFF;
    return STATUS.HELMET_OFF_IGNITION_OFF;
  };

  const currentStatus = getCurrentStatus();

  // ── RENDER ───────────────────────────────────
  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1a1a2e" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>🏍️ Smart Helmet System</Text>
        <View style={[styles.connDot,
          { backgroundColor: connected ? '#00C853' : '#D50000' }]} />
      </View>

      {/* Main Status Card */}
      <View style={[styles.statusCard, { borderColor: currentStatus.color }]}>
        <Text style={styles.statusIcon}>{currentStatus.icon}</Text>
        <Text style={[styles.statusText, { color: currentStatus.color }]}>
          {currentStatus.label}
        </Text>
        {confidence > 0 && (
          <Text style={styles.confidenceText}>
            Confidence: {(confidence * 100).toFixed(1)}%
          </Text>
        )}
      </View>

      {/* Stats Row */}
      <View style={styles.statsRow}>
        <StatBox
          label="Helmet"
          value={helmetDetected ? 'ON' : 'OFF'}
          color={helmetDetected ? '#00C853' : '#D50000'}
        />
        <StatBox
          label="Ignition"
          value={ignitionEnabled ? 'ON' : 'LOCKED'}
          color={ignitionEnabled ? '#00C853' : '#D50000'}
        />
        <StatBox
          label="Detections"
          value={detectionCount.toString()}
          color="#2196F3"
        />
      </View>

      {/* ESP32 Status */}
      {esp32Status && (
        <View style={styles.esp32Card}>
          <Text style={styles.sectionTitle}>📟 ESP32 Status</Text>
          <Text style={styles.esp32Text}>
            Relay: {esp32Status.ignition ? '🟢 ON' : '🔴 OFF'}  |
            Uptime: {esp32Status.uptime_s}s
          </Text>
        </View>
      )}

      {/* Last Update */}
      {lastUpdate && (
        <Text style={styles.lastUpdate}>Last update: {lastUpdate}</Text>
      )}

      {/* Event Log */}
      <Text style={styles.sectionTitle}>📋 Event Log</Text>
      <ScrollView style={styles.logContainer}>
        {log.map(entry => (
          <View key={entry.id} style={[styles.logEntry,
            { borderLeftColor: getLogColor(entry.type) }]}>
            <Text style={styles.logTime}>{entry.time}</Text>
            <Text style={styles.logMessage}>{entry.message}</Text>
          </View>
        ))}
        {log.length === 0 && (
          <Text style={styles.emptyLog}>No events yet...</Text>
        )}
      </ScrollView>

      {/* Reconnect Button */}
      {!connected && (
        <TouchableOpacity style={styles.reconnectBtn} onPress={connectMQTT}>
          <Text style={styles.reconnectText}>🔄 Reconnect</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

// ─────────────────────────────────────────────
// STAT BOX COMPONENT
// ─────────────────────────────────────────────
function StatBox({ label, value, color }) {
  return (
    <View style={styles.statBox}>
      <Text style={[styles.statValue, { color }]}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

function getLogColor(type) {
  const colors = {
    success: '#00C853',
    warning: '#FF6D00',
    error:   '#D50000',
    system:  '#2196F3',
    info:    '#607D8B',
  };
  return colors[type] || colors.info;
}

// ─────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'android' ? 30 : 50,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  connDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  statusCard: {
    backgroundColor: '#16213E',
    borderRadius: 16,
    borderWidth: 2,
    padding: 24,
    alignItems: 'center',
    marginBottom: 16,
  },
  statusIcon: {
    fontSize: 48,
    marginBottom: 8,
  },
  statusText: {
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  confidenceText: {
    color: '#90A4AE',
    marginTop: 6,
    fontSize: 13,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  statBox: {
    flex: 1,
    backgroundColor: '#16213E',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
    marginHorizontal: 4,
  },
  statValue: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  statLabel: {
    color: '#90A4AE',
    fontSize: 11,
    marginTop: 2,
  },
  esp32Card: {
    backgroundColor: '#16213E',
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
  },
  esp32Text: {
    color: '#B0BEC5',
    fontSize: 13,
    marginTop: 4,
  },
  sectionTitle: {
    color: '#90A4AE',
    fontSize: 13,
    fontWeight: '600',
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  lastUpdate: {
    color: '#607D8B',
    fontSize: 11,
    textAlign: 'right',
    marginBottom: 8,
  },
  logContainer: {
    flex: 1,
    backgroundColor: '#16213E',
    borderRadius: 12,
    padding: 8,
    marginBottom: 16,
  },
  logEntry: {
    borderLeftWidth: 3,
    paddingLeft: 10,
    paddingVertical: 4,
    marginBottom: 6,
  },
  logTime: {
    color: '#607D8B',
    fontSize: 10,
  },
  logMessage: {
    color: '#CFD8DC',
    fontSize: 13,
  },
  emptyLog: {
    color: '#607D8B',
    textAlign: 'center',
    padding: 20,
  },
  reconnectBtn: {
    backgroundColor: '#2196F3',
    borderRadius: 12,
    padding: 14,
    alignItems: 'center',
    marginBottom: 16,
  },
  reconnectText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
    fontSize: 15,
  },
});
