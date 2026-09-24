import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.joselofarias.kaggletpulab',
  appName: 'Kaggle TPU Lab ES',
  webDir: 'dist',
  server: {
    androidScheme: 'https',
    cleartext: true,
  },
  plugins: {
    CapacitorCookies: {
      enabled: false,
    },
    CapacitorHttp: {
      enabled: true,
    },
    LocalNotifications: {
      smallIcon: 'ic_launcher',
      iconColor: '#3b82f6',
    },
  },
};

export default config;
