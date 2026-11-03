import {test} from 'node:test';
import assert from 'node:assert/strict';
import {privateKeyToAccount,generatePrivateKey} from 'viem/accounts';
import {makeQuote,UBTC_ID,USDC_ID} from '../src/domain';
(globalThis as any).window=new EventTarget();
const {signOrder}=await import('../src/wallet');
const signer=privateKeyToAccount(generatePrivateKey()); // Ephemeral offline test identity, never funded or persisted.
const now=Date.now();
const quote=()=>makeQuote({market:{coin:'@142',asset:10142,base:{name:'UBTC',index:197,tokenId:UBTC_ID,szDecimals:5},quote:{name:'USDC',index:0,tokenId:USDC_ID,szDecimals:8}},book:{coin:'@142',time:Date.now(),levels:[[{px:'77500',sz:'1'}],[{px:'77510',sz:'1'}]]},account:signer.address,cloid:'0x00000000000000000000000000000001',amount:'25',available:'100',feeRate:'0.0007'});
test('single order signs only official L1 domain, recovers the connected wallet, no agent approval or exchange request',async()=>{
 const methods:string[]=[];let typed:any;
 const provider={async request({method,params}:any){methods.push(method);if(method==='eth_accounts')return [signer.address];if(method==='eth_chainId')return '0xa4b1';if(method==='eth_signTypedData_v4'){typed=JSON.parse(params[1]);return signer.signTypedData(typed);}throw new Error('Unexpected RPC '+method);}};
 const q=quote(),sig=await signOrder(provider,signer.address,q,now);
 assert.equal(sig.r.length,66);assert.equal(typed.domain.name,'Exchange');assert.equal(typed.domain.chainId,1337);assert.equal(typed.message.source,'a');assert.match(typed.message.connectionId,/^0x[0-9a-f]{64}$/);assert.deepEqual(new Set(methods),new Set(['eth_accounts','eth_chainId','eth_signTypedData_v4']));
});
test('user rejection and wrong signer never produce a submittable signature',async()=>{
 const bad={async request({method}:any){if(method==='eth_accounts')return [signer.address];if(method==='eth_chainId')return '0x1';throw new Error('User rejected request');}};
 await assert.rejects(signOrder(bad,signer.address,quote(),now),/rejected/);
 const other=privateKeyToAccount(generatePrivateKey());
 const wrong={async request({method,params}:any){if(method==='eth_accounts')return [signer.address];if(method==='eth_chainId')return '0x1';return other.signTypedData(JSON.parse(params[1]));}};
 await assert.rejects(signOrder(wrong,signer.address,quote(),now),/eşleşmiyor/);
});

test('spot action hash matches the official Python SDK, including nonce and expiry',async()=>{
 const {readFile}=await import('node:fs/promises');
 const {createL1ActionHash}=await import('@nktkas/hyperliquid/signing');
 const vector=JSON.parse(await readFile(new URL('./l1-vector.json',import.meta.url),'utf8'));
 assert.equal(createL1ActionHash({action:vector.action,nonce:vector.nonce,expiresAfter:vector.expiresAfter}),vector.expectedHash);
 assert.notEqual(createL1ActionHash({action:vector.action,nonce:vector.nonce+1,expiresAfter:vector.expiresAfter}),vector.expectedHash);
});
