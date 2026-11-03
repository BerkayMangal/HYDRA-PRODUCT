import { signL1Action } from '@nktkas/hyperliquid/signing';
import { recoverTypedDataAddress } from 'viem';
import { assertFresh, type Quote } from './domain';
export type Provider={request:(args:{method:string;params?:unknown[]})=>Promise<any>;on?:(event:string,listener:(value:any)=>void)=>void;removeListener?:(event:string,listener:(value:any)=>void)=>void;isMetaMask?:boolean;isRabby?:boolean};
export type WalletChoice={id:string;name:string;provider:Provider};
export const choices:WalletChoice[]=[];
function add(choice:WalletChoice){if(!choices.some(w=>w.provider===choice.provider))choices.push(choice);}
window.addEventListener('eip6963:announceProvider',((event:CustomEvent)=>{const d=event.detail;if(d?.provider?.request&&typeof d.info?.name==='string')add({id:d.info.uuid,name:d.info.name.slice(0,60),provider:d.provider});}) as EventListener);
export function discover(){const p=(window as any).ethereum;if(p?.request)add({id:'injected',name:p.isRabby?'Rabby':p.isMetaMask?'MetaMask':'Cüzdan',provider:p});return choices;}
export async function addressOf(provider:Provider):Promise<`0x${string}`>{
 const a=await provider.request({method:'eth_accounts'});if(!Array.isArray(a)||!/^0x[0-9a-fA-F]{40}$/.test(a[0]))throw new Error('Cüzdan bağlantısı bulunamadı.');return a[0];
}
export async function signOrder(provider:Provider,account:`0x${string}`,quote:Quote,nonce:number){
 assertFresh(quote,await addressOf(provider));const chain=await provider.request({method:'eth_chainId'});
 const wallet={
  getAddresses:async()=>[account],getChainId:async()=>Number(chain),
  async signTypedData(args:any):Promise<`0x${string}`>{
   const signature=await provider.request({method:'eth_signTypedData_v4',params:[account,JSON.stringify(args)]});
   const recovered=await recoverTypedDataAddress({...args,signature});
   if(recovered.toLowerCase()!==account.toLowerCase())throw new Error('İmza bağlı cüzdanla eşleşmiyor.');
   return signature;
  }
 };
 let signature;
 try { signature=await signL1Action({wallet,action:quote.action,nonce,isTestnet:false,expiresAfter:quote.expires}); }
 catch(error) { if(error instanceof Error && error.cause instanceof Error)throw error.cause; throw error; }
 assertFresh(quote,await addressOf(provider));
 if(await provider.request({method:'eth_chainId'})!==chain)throw new Error('İmza sırasında cüzdan ağı değişti. Yeniden fiyat al.');
 return signature;
}
