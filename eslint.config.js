import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';
export default tseslint.config({ignores:['**/node_modules/**','**/dist/**','TFT_NBEATS/**','LSTM_Light_GBM/**','Prophet_HistGradientBoosting/**']},...tseslint.configs.recommended,{files:['src/**/*.{ts,tsx}'],plugins:{'react-hooks':reactHooks},rules:{'@typescript-eslint/no-explicit-any':'off','@typescript-eslint/no-unused-vars':'off','react-hooks/rules-of-hooks':'error','react-hooks/exhaustive-deps':'warn'}});
