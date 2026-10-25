import {create} from 'zustand';
type State={theme:'dark'|'light';models:string[];common:boolean;toggleTheme:()=>void;setModels:(v:string[])=>void;setCommon:(v:boolean)=>void};
export const useStore=create<State>((set)=>({theme:(localStorage.getItem('theme') as 'dark'|'light')||'dark',models:[],common:true,toggleTheme:()=>set(s=>{const theme=s.theme==='dark'?'light':'dark';localStorage.setItem('theme',theme);return{theme}}),setModels:models=>set({models}),setCommon:common=>set({common})}));
