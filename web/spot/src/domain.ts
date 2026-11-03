import Decimal from 'decimal.js';

export const UBTC_ID = '0x8f254b963e8468305d409b33aa137c67';
export const USDC_ID = '0x6d1e7cde53ba9467b783cb7c530ce054';
export const QUOTE_TTL = 30_000;
export const API = 'https://api.hyperliquid.xyz';
export type Token = {name:string; index:number; tokenId:string; szDecimals:number};
export type Market = {coin:string; asset:number; base:Token; quote:Token};
export type Book = {coin:string; time:number; levels:{px:string; sz:string}[][]};
export type OrderAction = {type:'order'; orders:{a:number;b:true;p:string;s:string;r:false;t:{limit:{tif:'Ioc'}};c:`0x${string}`}[];grouping:'na'};
export type Quote = {created:number; expires:number; account:string; action:OrderAction; size:string; limit:string; maximum:string; estimated:string; feeReserve:string; feeRate:string; amount:string; cloid:`0x${string}`};

export function resolveMarket(meta:{tokens:Token[]; universe:{name:string; index:number; tokens:number[]}[]}):Market {
  const bases=meta.tokens.filter(t=>t.tokenId?.toLowerCase()===UBTC_ID);
  const quotes=meta.tokens.filter(t=>t.tokenId?.toLowerCase()===USDC_ID);
  if(bases.length!==1||quotes.length!==1)throw new Error('UBTC/USDC varlık kimliği doğrulanamadı.');
  const base=bases[0],quote=quotes[0];
  const markets=meta.universe.filter(m=>m.tokens.length===2&&m.tokens[0]===base.index&&m.tokens[1]===quote.index);
  if(markets.length!==1||!Number.isInteger(markets[0].index)||markets[0].index<0||base.szDecimals!==5)throw new Error('UBTC/USDC piyasası doğrulanamadı.');
  return {coin:markets[0].name,asset:10_000+markets[0].index,base,quote};
}
export function freeBalance(balances:{token:number; total:string; hold:string}[],token:number):string {
  const found=balances.filter(b=>b.token===token);
  if(found.length>1)throw new Error('Bakiye yanıtı tutarsız.');
  if(!found.length)return '0';
  const total=new Decimal(found[0].total),hold=new Decimal(found[0].hold);
  if(!total.isFinite()||!hold.isFinite()||total.isNegative()||hold.isNegative()||hold.gt(total))throw new Error('Bakiye doğrulanamadı.');
  return total.minus(hold).toFixed();
}
export function makeQuote({market,book,amount,available,feeRate,account,cloid,now=Date.now()}: {market:Market;book:Book;amount:string;available:string;feeRate:string;account:string;cloid:`0x${string}`;now?:number}):Quote {
  if(book.coin!==market.coin||!Number.isFinite(book.time)||now-book.time>10_000||book.time-now>5_000)throw new Error('Emir defteri güncel değil. Yeniden fiyat al.');
  if(!/^\d+(\.\d{1,2})?$/.test(amount))throw new Error('USDC tutarını en fazla iki ondalıkla gir.');
  const budget=new Decimal(amount),free=new Decimal(available),fee=new Decimal(feeRate);
  if(!budget.isFinite()||budget.lt(12)||budget.gt(1000))throw new Error('Bu ilk sürümde tutar 12–1.000 USDC arasında olmalı.');
  if(!fee.isFinite()||fee.lt(0)||fee.gt('0.01'))throw new Error('Spot komisyon oranı doğrulanamadı.');
  const bids=book.levels?.[0],asks=book.levels?.[1];
  if(!bids?.length||!asks?.length)throw new Error('Piyasada yeterli fiyat verisi yok.');
  const best=new Decimal(asks[0].px),bid=new Decimal(bids[0].px);
  if(!best.isFinite()||!bid.isFinite()||best.lte(0)||bid.lte(0)||bid.gt(best))throw new Error('Emir defteri tutarsız.');
  // Hard cap: 0.5% from the best ask. Spot prices have at most 5 significant figures
  // and 8-szDecimals fractional digits. Round down so the cap is never exceeded.
  const limit=best.mul('1.005').toSignificantDigits(5,Decimal.ROUND_DOWN).toDecimalPlaces(8-market.base.szDecimals,Decimal.ROUND_DOWN);
  const size=budget.div(limit).toDecimalPlaces(market.base.szDecimals,Decimal.ROUND_DOWN);
  const maximum=size.mul(limit),reserve=maximum.mul(fee),total=maximum.plus(reserve);
  if(size.lte(0)||size.mul(best).lt(10))throw new Error('Emir minimum piyasa tutarının altında.');
  if(total.gt(free))throw new Error('Spot USDC bakiyesi tutar ve komisyon payı için yetersiz.');
  let remaining=size,cost=new Decimal(0),previous=best;
  for(const level of asks){
    const px=new Decimal(level.px),sz=new Decimal(level.sz);
    if(!px.isFinite()||!sz.isFinite()||px.lt(previous)||sz.lte(0))throw new Error('Emir defteri tutarsız.');
    previous=px;if(px.gt(limit))break;
    const take=Decimal.min(remaining,sz);cost=cost.plus(take.mul(px));remaining=remaining.minus(take);if(remaining.isZero())break;
  }
  if(remaining.gt(0))throw new Error('Kayma sınırında yeterli likidite yok. Daha düşük tutar dene.');
  const action:OrderAction={type:'order',orders:[{a:market.asset,b:true,p:limit.toFixed(),s:size.toFixed(),r:false,t:{limit:{tif:'Ioc'}},c:cloid}],grouping:'na'};
  return {created:now,expires:now+QUOTE_TTL,account,action,size:size.toFixed(),limit:limit.toFixed(),maximum:maximum.toFixed(),estimated:cost.toFixed(),feeReserve:reserve.toFixed(),feeRate,amount,cloid};
}
export function assertFresh(q:Quote,account:string,now=Date.now()) {
  if(q.account.toLowerCase()!==account.toLowerCase()||now<q.created||now>=q.expires)throw new Error('Önizleme süresi doldu veya cüzdan değişti. Yeniden fiyat al.');
}
export function orderOutcome(response:any):{kind:'filled'|'unfilled'|'rejected';message:string} {
  if(response?.status!=='ok')return {kind:'rejected',message:String(response?.response??'Emir reddedildi.')};
  const statuses=response?.response?.data?.statuses;
  if(!Array.isArray(statuses)||statuses.length!==1)throw new Error('Emir sonucu belirsiz. Yeniden emir verme; Hyperliquid geçmişinden kontrol et.');
  const s=statuses[0];
  if(s.filled&&new Decimal(s.filled.totalSz).gt(0))return {kind:'filled',message:`Gerçekleşen: ${s.filled.totalSz} UBTC · Ortalama ${s.filled.avgPx} USDC · Emir #${s.filled.oid}. Kısmi gerçekleşme olabilir.`};
  if(s.error)return {kind:'rejected',message:String(s.error)};
  if(s==='success')return {kind:'unfilled',message:'Emir alındı; gerçekleşme doğrulanmadı. Geçmişi kontrol et.'};
  throw new Error('Emir sonucu belirsiz. Yeniden emir verme; Hyperliquid geçmişinden kontrol et.');
}
