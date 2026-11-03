import './style.css';
import Decimal from 'decimal.js';
import { info,post } from './api';
import { assertFresh,freeBalance,makeQuote,orderOutcome,resolveMarket,type Market,type Quote } from './domain';
import { addressOf,discover,signOrder,type Provider } from './wallet';

const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
const fmt=(v:string,places=2)=>new Decimal(v).toDecimalPlaces(places).toNumber().toLocaleString('tr-TR',{maximumFractionDigits:places});
let provider:Provider|undefined,account:`0x${string}`|undefined,market:Market|undefined,available='0',quote:Quote|undefined,busy=false,revision=0;
let detach:(()=>void)|undefined;
const pendingKey=(a:string)=>`hydra:spot:pending:${a.toLowerCase()}`;
const getPending=()=>account?localStorage.getItem(pendingKey(account)):null;
const app=$('app');
app.innerHTML=`<header><a class="brand" href="./" aria-label="HYDRA Spot">H<span>Y</span>DRA<i>SPOT</i></a><span class="network"><b></b> Hyperliquid · Mainnet</span></header>
<main><div class="intro"><p class="eyebrow">CÜZDANIN SENDE. KARAR SENDE.</p><h1>Bitcoin’e bir adım.</h1><p>Spot USDC bakiyenle BTC (UBTC) al.<br>Her emir, kendi cüzdanında vereceğin imzayla gönderilir.</p></div>
<div class="layout"><section class="trade card"><div class="card-head"><div class="asset-icon">₿</div><div><h2>BTC <span>UBTC</span></h2><p>Unit Bitcoin / USDC</p></div><span class="tag">SPOT</span></div>
<div class="wallet-row"><span id="wallet-address">Cüzdan bağlı değil</span><button id="disconnect" class="text" hidden>Bağlantıyı kes</button></div>
<div id="wallet-options"><label for="wallet-select">Cüzdan</label><select id="wallet-select"></select><button id="connect" class="primary">Cüzdanı bağla</button></div>
<div id="mobile-help" hidden><p>Telefonda bu sayfayı cüzdanının tarayıcısında aç.</p><a id="metamask-link" class="button primary">MetaMask’te aç ↗</a><p class="small">Rabby veya başka bir cüzdan kullanıyorsan bu sayfanın adresini cüzdan tarayıcısına yapıştır.</p></div>
<div class="balance"><span>Kullanılabilir spot USDC</span><strong id="balance">—</strong></div>
<label for="amount">Alım tutarı</label><div class="amount-wrap"><input id="amount" type="text" inputmode="decimal" autocomplete="off" placeholder="25" value="25" aria-describedby="amount-note"><span>USDC</span></div>
<p class="small" id="amount-note">12–1.000 USDC · Komisyon payı bakiyede ayrıca bulunmalı.</p>
<div class="presets"><button data-amount="25">25 USDC</button><button data-amount="50">50 USDC</button><button data-amount="100">100 USDC</button></div>
<button id="preview" class="primary" disabled>Emri incele <span>→</span></button>
<div id="quote" hidden><div class="divider"></div><div class="quote-heading"><h3>Alım önizlemesi</h3><span id="expires"></span></div><dl><div><dt>Talep edilen UBTC</dt><dd id="quote-size"></dd></div><div><dt>Tahmini işlem tutarı</dt><dd id="quote-estimated"></dd></div><div><dt>Azami birim fiyat</dt><dd id="quote-limit"></dd></div><div><dt>Azami işlem tutarı</dt><dd id="quote-maximum"></dd></div><div><dt>Komisyon için ayrılan pay</dt><dd id="quote-fee"></dd></div></dl><p class="small">Kayma sınırı %0,5. Emir hemen gerçekleşen kısmı alır; kalan miktar iptal olur (IOC). Komisyon, gerçekleşen işleme göre değişebilir ve UBTC’den kesilebilir.</p><button id="buy" class="primary">Cüzdanda imzala ve al</button><details><summary>İmzalanacak emir içeriği</summary><pre id="payload"></pre><p class="small">Hyperliquid cüzdanda “Exchange / Agent” ve bir özet değer gösterir. Bu tek emrin imzasıdır; API cüzdanı yetkilendirmesi değildir. İmza alanının zincir kimliği 1337’dir. Cüzdanın bunu desteklemiyorsa emir gönderilmez.</p></details></div>
<p id="status" role="status" aria-live="polite">Cüzdan bağlamak işlem başlatmaz.</p><button id="reconcile" class="secondary" hidden>Son emrin durumunu kontrol et</button>
</section><aside><section class="card guide"><p class="eyebrow">İLK DENEMEN</p><h2>Üç adımda hazır.</h2><ol><li><span>01</span><div><strong>Cüzdanını bağla</strong><p>Telefonda MetaMask veya Rabby’nin uygulama içi tarayıcısını kullan.</p></div></li><li><span>02</span><div><strong>Spot bakiyeni kontrol et</strong><p>Bu ekran Hyperliquid hesabındaki spot USDC’yi kullanır. Cüzdanındaki diğer ağ bakiyeleri burada harcanmaz.</p></div></li><li><span>03</span><div><strong>Önizle, sonra imzala</strong><p>Tutarı ve fiyat sınırını incele. Gerçek alım yalnızca “imzala ve al” adımından sonra yapılır.</p></div></li></ol><a class="external" href="https://app.hyperliquid.xyz/" target="_blank" rel="noopener noreferrer">Hyperliquid hesabını aç ↗</a></section><section class="asset-note"><span>UBTC HAKKINDA</span><p>UBTC, Unit üzerinden temsil edilen Bitcoin varlığıdır. Bitcoin ağına çekim ayrı bir işlemdir; bu ekran çekim veya para yatırma yapmaz.</p></section><section class="history"><div class="history-head"><h3>Son spot işlemler</h3><button id="refresh" class="text" disabled>Yenile ↻</button></div><p id="history">Bağlantıdan sonra burada görünür.</p></section></aside></div></main><footer>HYDRA · Spot erken erişim <span>Varlıklar senin Hyperliquid hesabında kalır.</span></footer>`;

function status(message:string,error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
function invalidate(){quote=undefined;$('quote').hidden=true;}
function update(){
 const connected=!!account;$('wallet-options').hidden=connected||!discover().length;
 $('mobile-help').hidden=connected||!!discover().length;
 $('disconnect').hidden=!connected;($('disconnect') as HTMLButtonElement).disabled=busy;
 $('wallet-address').textContent=account?`${account.slice(0,6)}…${account.slice(-4)}`:'Cüzdan bağlı değil';
 $('wallet-address').title=account??'';
 ($('preview') as HTMLButtonElement).disabled=busy||!account||!market||!!getPending();
 ($('refresh') as HTMLButtonElement).disabled=busy||!account;
 ($('amount') as HTMLInputElement).disabled=busy;
 ($('buy') as HTMLButtonElement).disabled=busy||!quote||Date.now()>=quote.expires||!!getPending();
 $('reconcile').hidden=!getPending();($('reconcile') as HTMLButtonElement).disabled=busy;
}
function walletChoices(){const old=($('wallet-select') as HTMLSelectElement).value;$('wallet-select').replaceChildren(...discover().map(w=>{const o=document.createElement('option');o.value=w.id;o.textContent=w.name;return o;}));if(discover().some(w=>w.id===old))($('wallet-select') as HTMLSelectElement).value=old;update();}
const launch=new URL(location.href);launch.hash='';launch.search='';($('metamask-link') as HTMLAnchorElement).href=`https://metamask.app.link/dapp/${launch.host}${launch.pathname}`;
async function refresh(){
 if(!account||!market)return;
 const a=account,r=revision;
 const [state,fills]=await Promise.all([info({type:'spotClearinghouseState',user:a}),info({type:'userFills',user:a,aggregateByTime:true})]);
 if(a!==account||r!==revision)return;
 available=freeBalance(state.balances,market.quote.index);$('balance').textContent=`${fmt(available)} USDC`;
 const recent=fills.filter((f:any)=>f.coin===market!.coin).slice(0,5);
 const holder=$('history');holder.replaceChildren();
 if(!recent.length)holder.textContent='Bu hesapta UBTC/USDC işlemi görünmüyor.';
 else for(const f of recent){const row=document.createElement('span');row.className='fill';row.textContent=`${f.side==='B'?'Alım':'Satım'} · ${f.sz} UBTC · ${fmt(f.px)} USDC · ${new Date(f.time).toLocaleString('tr-TR')}`;holder.append(row);}
}
function reset(){revision++;detach?.();detach=undefined;provider=undefined;account=undefined;market=undefined;available='0';invalidate();$('balance').textContent='—';$('history').textContent='Bağlantıdan sonra burada görünür.';status('Cüzdan bağlantısı kesildi.');update();}
$('disconnect').onclick=reset;
$('connect').onclick=async()=>{
 if(busy)return;busy=true;update();
 try{
  provider=discover().find(w=>w.id===($('wallet-select') as HTMLSelectElement).value)?.provider;if(!provider)throw new Error('Cüzdan bulunamadı. Sayfayı cüzdan tarayıcısında aç.');
  await provider.request({method:'eth_requestAccounts'});account=await addressOf(provider);revision++;
  const p=provider;const changed=()=>{reset();status('Cüzdan veya ağ değişti. Tekrar bağlan.');};p.on?.('accountsChanged',changed);p.on?.('chainChanged',changed);p.on?.('disconnect',changed);detach=()=>{p.removeListener?.('accountsChanged',changed);p.removeListener?.('chainChanged',changed);p.removeListener?.('disconnect',changed);};
  market=resolveMarket(await info({type:'spotMeta'}));await refresh();status(getPending()?'Önce son emrin durumunu kontrol et.':new Decimal(available).gt(0)?'Cüzdan bağlı. Tutarı girerek emir önizlemesini açabilirsin.':'Cüzdan bağlı. Alım için Hyperliquid hesabında spot USDC gerekiyor.');
 }catch(e){status(errorText(e),true);}finally{busy=false;update();}
};
$('amount').oninput=()=>{invalidate();update();};
for(const b of document.querySelectorAll<HTMLButtonElement>('[data-amount]'))b.onclick=()=>{if(busy)return;($('amount') as HTMLInputElement).value=b.dataset.amount!;invalidate();update();};
function errorText(e:unknown){const text=e instanceof Error?e.message:String(e);if(/chain.?id|active chain|network mismatch/i.test(text))return 'Cüzdan, Hyperliquid’in 1337 imza alanını desteklemiyor. Emir gönderilmedi. Bu cüzdanda bağlantı ve önizleme kullanılabilir.';if(/reject|denied|4001/i.test(text))return 'Cüzdanda işlem iptal edildi. Emir gönderilmedi.';return text.slice(0,400);}
$('preview').onclick=async()=>{
 if(busy||!provider||!account||!market||getPending())return;
 busy=true;invalidate();update();status('Güncel bakiye, komisyon ve fiyat okunuyor…');
 const a=account,r=revision,m=market;
 try{
  const [book,state,fees]=await Promise.all([info({type:'l2Book',coin:m.coin}),info({type:'spotClearinghouseState',user:a}),info({type:'userFees',user:a})]);
  if(r!==revision||a!==account)throw new Error('Cüzdan değişti.');
  if((await addressOf(provider)).toLowerCase()!==a.toLowerCase())throw new Error('Cüzdan değişti.');
  available=freeBalance(state.balances,m.quote.index);$('balance').textContent=`${fmt(available)} USDC`;
  const random=crypto.getRandomValues(new Uint8Array(16));const cloid=`0x${Array.from(random,x=>x.toString(16).padStart(2,'0')).join('')}` as `0x${string}`;
  quote=makeQuote({market:m,book,amount:($('amount') as HTMLInputElement).value.trim().replace(',','.'),available,feeRate:fees.userSpotCrossRate,account:a,cloid});
  $('quote-size').textContent=`${quote.size} UBTC`;$('quote-estimated').textContent=`≈ ${fmt(quote.estimated)} USDC`;$('quote-limit').textContent=`${fmt(quote.limit)} USDC`;$('quote-maximum').textContent=`${fmt(quote.maximum)} USDC`;$('quote-fee').textContent=`≤ ${fmt(quote.feeReserve,4)} USDC (%${new Decimal(quote.feeRate).mul(100).toFixed()})`;$('payload').textContent=JSON.stringify(quote.action,null,2);$('quote').hidden=false;status('Önizlemeyi incele. Henüz emir gönderilmedi.');
 }catch(e){invalidate();status(errorText(e),true);}finally{busy=false;update();}
};
$('buy').onclick=async()=>{
 if(busy||!provider||!account||!market||!quote||getPending())return;
 busy=true;update();const q=structuredClone(quote),p=provider,a=account,r=revision;
 let attempted=false;
 try{
  assertFresh(q,a);status('Bu tek alım emrini cüzdanında incele ve imzala…');const nonce=Date.now();
  const signature=await signOrder(p,a,q,nonce);
  if(revision!==r||account!==a)throw new Error('Cüzdan değişti. Emir gönderilmedi.');assertFresh(q,a);
  // Recheck identity and available balance after the user signs, before submission.
  const [meta,state]=await Promise.all([info({type:'spotMeta'}),info({type:'spotClearinghouseState',user:a})]);
  const current=resolveMarket(meta);if(current.asset!==q.action.orders[0].a)throw new Error('Piyasa kimliği değişti. Emir gönderilmedi.');
  if(new Decimal(freeBalance(state.balances,current.quote.index)).lt(new Decimal(q.maximum).plus(q.feeReserve)))throw new Error('Spot bakiye değişti. Emir gönderilmedi.');
  assertFresh(q,await addressOf(p));if(revision!==r)throw new Error('Cüzdan değişti. Emir gönderilmedi.');
  localStorage.setItem(pendingKey(a),JSON.stringify({cloid:q.cloid,time:Date.now()}));attempted=true;invalidate();status('Emir gönderiliyor. Sonuç doğrulanmadan yeni emir gönderme.');
  const response=await post('exchange',{action:q.action,signature,nonce,expiresAfter:q.expires});
  const result=orderOutcome(response);localStorage.removeItem(pendingKey(a));attempted=false;
  if(account===a){status(result.message,result.kind==='rejected');try{await refresh();}catch{status(result.message+' Bakiye yenilemesi başarısız; geçmişi Hyperliquid üzerinden kontrol et.',result.kind==='rejected');}}
 }catch(e){if(attempted)status('Sonuç henüz doğrulanamadı. Yeniden alım yapma; aşağıdan emrin durumunu kontrol et.',true);else status(errorText(e),true);invalidate();}finally{busy=false;update();}
};
$('reconcile').onclick=async()=>{
 if(busy||!account||!getPending())return;const a=account;busy=true;update();
 try{
  const pending=JSON.parse(getPending()!);const result=await info({type:'orderStatus',user:a,oid:pending.cloid});
  if(result.status==='order'){
   const s=result.order?.status;
   if(s==='filled'||/canceled|rejected/i.test(s)){localStorage.removeItem(pendingKey(a));status(`Son emir: ${s}. Gerçekleşen miktar için işlem geçmişini kontrol et.`);await refresh();}
   else status(`Son emir durumu: ${s??'belirsiz'}. Yeni alım kapalı. Hyperliquid üzerinden kontrol et.`,true);
  }else if(result.status==='unknownOid'&&Date.now()-pending.time>120_000){localStorage.removeItem(pendingKey(a));status('Hyperliquid bu emir kimliğini bulamadı. Tekrar almadan önce Hyperliquid işlem geçmişini de kontrol et.');await refresh();}
  else status('Emir sonucu henüz kesinleşmedi. Biraz sonra tekrar kontrol et.',true);
 }catch(e){status(errorText(e),true);}finally{busy=false;update();}
};
$('refresh').onclick=async()=>{if(busy)return;busy=true;invalidate();update();try{await refresh();status('Bakiyeler ve spot işlem geçmişi güncellendi.');}catch(e){status(errorText(e),true);}finally{busy=false;update();}};
setInterval(()=>{if(quote){const remaining=Math.max(0,Math.ceil((quote.expires-Date.now())/1000));$('expires').textContent=remaining?`${remaining} sn`:'Süre doldu';}update();},1000);
walletChoices();window.addEventListener('eip6963:announceProvider',walletChoices);window.dispatchEvent(new Event('eip6963:requestProvider'));setTimeout(walletChoices,500);
