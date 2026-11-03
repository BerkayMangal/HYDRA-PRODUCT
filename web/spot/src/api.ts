import { API } from './domain';
export async function post(path:'info'|'exchange',body:unknown):Promise<any>{
  const response=await fetch(`${API}/${path}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(15_000),credentials:'omit',cache:'no-store'});
  if(!response.ok)throw new Error(`Hyperliquid bağlantısı başarısız (${response.status}).`);
  return response.json();
}
export const info=(body:unknown)=>post('info',body);
