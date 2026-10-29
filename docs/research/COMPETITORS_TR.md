> Tarihli araştırma/audit kaydıdır. Güncel kod, bulgular ve çalışma durumu için [29 Ekim teslimini](FINAL_TR.md) ve [çalıştırma kılavuzunu](RUNBOOK_TR.md) okuyun. Bu kayıttaki eski hata ve süreç durumları bugünkü durumun beyanı değildir.

# Hyperliquid, likidasyonlar ve rakip araştırması

28 Ekim 2026. İlk API gözlemleri yaklaşık 12:25–12:39 Türkiye saati.

## Sonuç

**Araştırılabilir yeni veri var; doğrulanmış yeni alfa yok.** Public Hyperliquid
API'sinden cüzdan pozisyonları, tahmini likidasyon seviyeleri ve açıkça
likidasyon olarak işaretlenen gerçekleşmiş fill'ler alınabildi. API anahtarı
gerekmedi. Bunların hangisinin gelecekteki fiyat hareketine ek bilgi kattığı
henüz gösterilmiş değil.

En öncelikli hipotez, gerçekleşmiş zorunlu akış sonrasında **spot absorpsiyonu
ile devam/geri dönüşü ayırmak**. İkinci hipotez, sabit bir cüzdan örnekleminin
gerçek baz miktar değişiminin, fiyat trendine ek bilgi taşıması. Mevcut
likidasyon fiyatlarını geçmiş fiyat grafiğine koyup backtest yapmak geçersiz.

## Rakip/ekosistem incelemesi

Doğrudan sayfa erişimi ile dokümandan doğrulanan özellikler ayrıdır. Aşağıdaki
okumalar bir ürünü kullanarak performans testi yaptığımız anlamına gelmez.

| Kaynak | Gerçekten doğrulanan | HYDRA için alınacak fikir | Sınır |
|---|---|---|---|
| **Nansen** resmî açık CLI ve referans dokümanları | `position-intelligence` ile trader cohort bazında pozisyon; wallet profiler ve perp pozisyon/PnL komutları; adres etiketleri, leverage, margin türü, liquidation price, funding, uPnL alanları | Adres listesinden ziyade cohort + pozisyon değişimi + zaman damgası görünümü | Veri path'i kullanılmadı. Auth/session, plan ve credit kavramları dokümanda var; güncel ücret veya tüm özelliklerin ücretsiz olduğu doğrulanmadı |
| **Hypurrscan_Monitor**, BugSmith'in üçüncü taraf açık projesi | README: 2 dakikalık pozisyon yenileme, yeni pozisyon ve %10 değer değişimi alarmı, çoklu adres | Basit watchlist ve değişiklik günlüğü | HypurrScan'in resmî ürünü değildir. Resmî HypurrScan arayüzü doğrudan açılamadı. README'deki “insider” etiketi kanıt sayılmadı |
| **hyperdash-tracker**, kittleik'in bağımsız açık projesi | Public API ile günlük hesap/fill snapshot'ı ve günler arası rapor | Tekrarlanabilir günlük kayıt ve değişim raporu | Ticari hyperdash.com ile aynı ürün olduğu varsayılmadı; kaynak README incelemesi, çalıştırma yapılmadı |
| **Pain / HomeDeck** | İstenen URL'ye bağlantı proxy 403 ile engellendi | Likidasyon/pozisyon görselleştirmesi için kullanıcının işaret ettiği inceleme hedefi | Arayüzü, veri kaynağı, kapsamı, hesaplama yöntemi ve ücretleri doğrulanmadı |
| **CoinGlass, Hyblock, CoinAnk, hyperdash.com** | Doğrudan erişim girişimleri proxy 403 aldı | Pazar haritasındaki diğer inceleme hedefleri | Güncel özellik, kapsam, gecikme veya ücret karşılaştırması tamamlanamadı; bunlar doğrulanmış rakip özellikleri diye raporlanmadı |

**Nansen'den önemli ders:** “All-time profitable”, “30D Smart Trader” ve
“Smart HL Perps Trader” gibi etiketler tanımlanıyor. Bugünkü etiketli cüzdanları
geçmişe taşıyarak test yapmak seçim/sağ kalan yanlılığı üretir. Etiketin o gün
erişilebilir olduğu tarih veya ileriye dönük dondurulmuş cohort gerekir.

**Watchlist araçlarından önemli ders:** Dolar pozisyon değeri yükseldi diye
yeni alım yapılmış sayılmaz; yalnız fiyat yükselmiş olabilir. Biz değişimi
`szi` baz miktarından hesaplıyoruz. Para yatırma, PnL ve notional değişimi
aynı olay değildir. Açık uPnL de kapatılmış, maliyet sonrası strateji başarısı
anlamına gelmez.

## HYDRA'nın farklılaşabileceği yer

1. Tahmini likidasyon envanteri ile gerçekleşmiş likidasyon fill'ini ayrı göstermek.
2. Her rakamda kaynak/receipt zamanı, gecikme, cüzdan sayısı ve örneklem kapsamı.
3. En büyük hesabın ağırlığı ve pozisyonların kaç bağımsız hesaba yayıldığı.
4. Likidasyon/fiyat hareketini spot akışı ve gerçek envanter değişimiyle karşılaştırmak.
5. Her alarmın sonraki 15m/1h/4h sonucunu, maliyet ve belirsizlikle kaydetmek.

Ham public pozisyon verisi tek başına özgün bilgi avantajı değildir. Güvenilir
geçmiş/erişim zamanı, örneklem kalitesi ve gelecekte doğrulanan karar katkısı
daha değerli. İki dakikalık polling ile milisaniyelik execution avantajı iddia
edilmiyor. Akış örneklemi tüm piyasanın likidasyonlarını kapsamaz.

## Mevcut kodda bulunan ve düzeltilen hatalar

`collectors/venom_collector.py` Hyperliquid `openInterest` miktarını dolar
sanıyordu. Yaklaşık 42.723 BTC OI'ı, `$1M` filtresinde küçük sayıp BTC'yi
eliyordu. O anki mark fiyatıyla doğru notional yaklaşık **$3,48 milyar**.
Artık baz miktar ile dolar notional ayrı; filtre ve `oi_m` dolar kullanıyor.

Funding geçmişinde ardışık ödemelerin yaklaşık bir saat arayla geldiği
doğrulandı. Önceki kod `funding * 365` kullanıyordu; saatlik oran için basit
yıllıklandırma `funding * 24 * 365`. Örnekte saatlik `0.0000125`, sabit kaldığı
varsayımıyla **%10,95 basit APR** demek. Bu APY veya gelecekte alınacağı
garanti edilen getiri değildir. Sağlık kontrolünün bağlantı hatasında `True`
döndürmesi de düzeltildi.

## Public API ve cüzdan örneklemi

İlk pilotte BTC recentTrades içindeki 10 işlemin katılımcılarından 11 benzersiz
adres alındı. Bunlar rastgele piyasa örneklemi veya “smart money” değildi.
İlk pilot BTC gross pozisyonu yalnız yaklaşık $244K olduğundan büyük cüzdan
araştırması için ayrı bir keşif yapıldı.

Keşif sözleşmesi API hesaplarını okumadan önce kaydedildi:

- BTC, ETH, SOL ve HYPE public recentTrades katılımcıları.
- İlk 48 lexicographic benzersiz adresin public hesap durumunu inceleme.
- Default perp DEX gross açık pozisyonu en az $1M olanlardan, o anki büyüklükle
  en çok 12 adres seçme. PnL veya sonradan gerçekleşen getiri kullanılmadı.
- **6 hesap bulundu; bu cohort donduruldu.** Sonuç kötüleşince hesaptan
  çıkarılmayacak, yeni kazananlar sessizce eklenmeyecek.

İlk büyük hesap snapshot'ında:

| Ölçü | Gözlenen |
|---|---:|
| Başarılı hesap | 6 / 6 |
| Aktif pozisyon satırı, default DEX | 234 |
| Örneklem BTC gross notional | Yaklaşık $49,0M |
| BTC notional'ında en büyük hesabın payı | %81,3 |
| BTC gross ağırlıklarına göre etkin hesap sayısı | Yaklaşık 1,48 |
| Tahmini likidasyon fiyatı mark'a %2'den yakın BTC envanteri | Bu snapshot'ta 0 |

ETH tarafında tek hesabın ağırlığı yaklaşık %97,5. Bu sayıların “büyük para
topluca şu yöne gidiyor” diye okunması yanlış. Hesap market maker veya başka
yerde hedge edilmiş olabilir; gerçek sahibi veya diğer hesapları belirlenmedi.
Gross position büyüklüğü hesap özsermayesi değildir.

Kapsam yalnız seçilen default-DEX hesaplar. Başka cüzdanlar, subaccount'lar,
spot hesapları, HIP-3 DEX'ler ve dış borsalardaki hedge'ler tam kapsanmıyor.
Missing liquidation price risksiz pozisyon anlamına gelmez. Cross margin,
teminat ekleme/çekme ve diğer pozisyonlar likidasyon seviyesini değiştirebilir.

## Gerçek likidasyon kanıtı

Resmî SDK'nın genel fill şemasından farklı olarak gerçek yanıtlarda optional
`liquidation` nesnesi görüldü: `liquidatedUser`, `markPx`, `method`.

- İzlenen cüzdan victim ise fill side A, long pozisyonun zorunlu satışını;
  side B, short pozisyonun zorunlu alımını gösterir.
- İzlenen cüzdan karşı taraf ise bu yön ters çevrilir.
- Aynı fill iki hesabın yanıtında görünürse `(coin, tid)` ile bir kez sayılır.
- Sıradan `Close Long` / `Close Short`, explicit liquidation kanıtı olmadan
  zorunlu kapatma sayılmaz. Backstop ile market yöntemi ayrıca korunur.

İlk pilotte 6 explicit liquidation fill'i bulundu. Bunların **4'ü BTC** idi;
aynı piyasa dakikası/yön örnekleri birleştirilince **3 ayrı piyasa anı** kaldı.
Tam piyasa likidasyon hacmi veya her victim'in bütün pozisyonu değildir;
örneklem karşı tarafı yalnız bir parçayı görmüş olabilir.

## Küçük geçmiş olay etüdü

Hyperliquid'den 8 günlük yaklaşık 2.305 adet 5m BTC mumu alındı. Kapanmış
mumlar kullanıldı; giriş referansı olaydan sonraki 5m açılışı.

| BTC olay grubu | Likide yön | Gözlenen fill notional | 15m ters yön brüt fiyat hareketi | 1h | 4h |
|---|---|---:|---:|---:|---:|
| 1 | Long, 2 victim | $398,50 | -%0,343 | +%0,109 | +%1,074 |
| 2 | Long | $1.220,03 | +%0,203 | +%0,482 | -%0,060 |
| 3 | Short | $290,56 | +%0,191 | Henüz veri yok | Henüz veri yok |

**Bu, geçerli OOS backtest veya net strateji getirisi değildir.** Cüzdanlar
bugünkü aktiviteden seçildi; geçmiş fill'ler bugün çekildi. Birkaç küçük olayın
öncesinde/sonrasında erişilebilir olan bilgiyi kanıtlamıyor. Funding, gerçek
fill, slippage ve portföy simülasyonu yapılmadı. 24 bp round-trip maliyet
varsayımıyla bazı küçük brüt hareketler zaten ekonomik avantaj bırakmıyor.

İlk sonuç “likidasyondan sonra her zaman ters yöne gir” kuralını desteklemiyor:
aynı olayın 15m, 1h ve 4h yönleri bile farklılaşabiliyor. Örneklem herhangi
bir kuralı kabul veya reddetmek için çok küçük.

## Deneye dönüşecek hipotezler

**H1 — Likidasyon + spot absorpsiyonu.** Yalnız olay anından önce bilinen
değişkenlerle: zorunlu long satışı sonrası spot alım desteği, fiyatın olay
bölgesini geri almasıyla birlikte gelecekteki dönüşe bilgi ekliyor mu?
Tersi short likidasyonu için. Spot akışı ters yönde kuvvetliyse devam senaryosu
ayrı baseline. Basit “likidasyon oldu” ve yalnız fiyat/volatilite kontrolü şart.

**H2 — Örneklem envanter değişimi.** Aynı dondurulmuş cüzdanların signed baz
pozisyonu yaklaşık 15 dakikada nasıl değişti? Fiyat etkisi çıkarılmış halde,
eşit hesap yönü ile notional ağırlıklı yön aynı mı? En büyük hesabı çıkarınca
bulgu kayboluyor mu? Kullanılan pencerenin gerçek süresi ayrıca kayıtlı.

**H3 — Tahmini risk seviyesinin değişimi.** Pozisyon büyümeden liquidation
seviyesi değişiyorsa bu akış yerine margin değişimi olabilir. İki olay
birbirine karıştırılmamalı. Bugünkü tahmini seviyeler geçmişe taşınmamalı.

Başlangıçta üç farklı araştırma ailesi; sonuç gördükçe yüzlerce eşik üretme
yok. İleriye dönük kayıt, olay kümelerinin tekilleştirilmesi, as-of replay,
farklı rejimler, fee/funding/slippage ve çoklu deneme hesabı gerekir. Az sayıda
işlemden kârlılık açıklaması yapılmayacak. Daha geniş market-wide liquidation
coverage gerekirse önce bağımsız veri kaynağının artı katkısı ölçülmeli.

## Çalışan bileşenler

- Salt okunur public istemci: repo `collectors/hyperliquid_research.py`.
  Yalnız izinli `/info` türleri; signer/private key/exchange order API yok.
- Özellikler ve kurallar: repo `research/hyperliquid_features.py`.
- Cüzdan/pozisyon/fill journal: `collector.py`, `observations.sqlite3`.
- Dondurulmuş adres örneklemi: `large-wallet-cohort.json`; seçim sözleşmesi
  `cohort-discovery-spec.json`, kaynak ham verileri aynı dizinde.
- Metaveri, state ve liquidation kayıtlarında kaynak/ilk receipt zamanı
  korunuyor. Kaynak zamanı olmayan metadata için receipt zamanı açık.
- Poll aralığı 120 saniye, en çok 12 hesap, düşük disk alanında duruş.
- Default-DEX boş hesap gerçek flat; başarısız/eski API yanıtı flat değil unknown.
- 15 dakikalık delta için yeterli eski snapshot yoksa değer missing kalıyor.
- `userFillsByTime` 2.000 satıra ulaşırsa kesilme riski işaretleniyor; geçmişin
  eksiksiz olduğu veya bütün piyasanın izlendiği iddia edilmiyor.

**98 uygulama testi geçti**; OI/funding birimleri, proxy hata durumu, liquidation
victim/counterparty yönü, çift sayım, eski/gelecek veri, cohort değişimi ve public
endpoint sınırı kapsandı. Gerçek API denemesinde 6/6 hesap okunabildi.

Cüzdan gözlem süreci mevcut cloud makinesinde başlatıldı. Hedef duruş
**29 Ekim 2026 08:00 Türkiye saati**; makine çalıştığı sürece geçerli. Mevcut
OKX/EMA gölge worker'ı ayrıca çalışıyor. Railway değiştirilmedi; yeni cüzdan
özellikleri üretim karar motoruna bağlanmadı, emir/bildirim gönderilmedi.

## Erişimde kalan iş

Pain, HypurrScan, Hyperdash, Nansen web sayfası, CoinGlass, Hyblock, CoinAnk ve
Hyperliquid GitBook/statistics sayfaları için gereken alan adları eski custom
izinler korunarak **ağ ayarları taslağına kaydedildi**. Kaydetmek çalışma
oturumuna otomatik erişim veya yayın sağlamadı. Doğrudan ürün arayüzü ve güncel
fiyatlandırma araştırmasını tamamlamak için ortam ayarlarında taslak değişikliği
kaydedilmeli ve ortam yayımlanmalı; ardından engellenen istekler tekrar denenmeli.

Bu nedenle Pain'in birebir ekran/özellik incelemesi ve ticari araçların güncel
plan karşılaştırması henüz tamamlanmış sayılmıyor.

## Kaynaklar

- [Hyperliquid resmî Python SDK Info; incelenen commit](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/2fdb18f9517675ea03695a0962bd19eece9c83f0/hyperliquid/info.py).
  Commit'teki dosyayla indirilen dosyanın aynı olduğu kontrol edildi.
- [Public API](https://api.hyperliquid.xyz/info): `metaAndAssetCtxs`, `recentTrades`,
  `clearinghouseState`, `userFillsByTime`, `userNonFundingLedgerUpdates`,
  `fundingHistory`, `candleSnapshot`; ham yanıtlar ve observed zamanları saklandı.
- [Nansen CLI resmî açık README](https://github.com/nansen-ai/nansen-cli).
- [Nansen komut referansı](https://github.com/nansen-ai/nansen-skills/blob/main/references/commands.md).
- [Nansen Smart Money etiket tanımları](https://github.com/nansen-ai/nansen-skills/blob/main/references/smart-money-labels.md).
- [Üçüncü taraf Hypurrscan_Monitor](https://github.com/BugSmith/Hypurrscan_Monitor).
- [Bağımsız günlük tracker](https://github.com/kittleik/hyperdash-tracker).
- [Pain; doğrudan erişim engelli](https://pain.homedeck.xyz/).

Tekrar üretim: repo testleri `python -m pytest tests/ -q`; bu dizindeki
`event_study.py` yalnız dondurulmuş küçük pilotu tekrar özetler.
`collector.py --once` bir gerçek public API gözlemi ekler; eski örneklem
dosyaları veya seçim sözleşmesi sonuçlara bakılarak değiştirilmemeli.
