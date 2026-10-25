import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
  plugins:[react()],
  server:{proxy:{'/api':'http://127.0.0.1:8000'}},
  build:{rollupOptions:{output:{manualChunks(id){
    if(id.indexOf('/node_modules/recharts/')>=0)return 'charts';
    if(id.indexOf('/node_modules/lucide-react/')>=0)return 'icons';
    if(id.indexOf('/node_modules/framer-motion/')>=0)return 'motion';
    if(id.indexOf('/node_modules/react/')>=0||id.indexOf('/node_modules/react-dom/')>=0||id.indexOf('/node_modules/scheduler/')>=0)return 'react-vendor';
    if(id.indexOf('/node_modules/')>=0)return 'vendor';
  }}}}
});
