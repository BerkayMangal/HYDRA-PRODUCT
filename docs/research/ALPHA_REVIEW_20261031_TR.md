# 31 Ekim: masraf ayrıştırması ve yeni alfa kaynakları

**Doğrulanmış alfa bulunmadı.** Bu devam çalışması, ilk TWAP işlemlerinin
ekonomik sonucunu açıklıyor, yeni aktif hesap kaynağını tarıyor ve fonlama
ödemelerinin maliyet eşiğini ölçüyor. İleri testin dondurulmuş kuralları,
hesapları ve maliyetleri değiştirilmedi. Sonuç kötüleşti diye eski kayıtlar
silinmedi veya yeni bir eşikle geçmişten kâr üretilmedi.

## İlk model işlemlerinde ne görüldü?

31 Ekim **11:36 UTC** incelemesinde iki tamamlanmış TWAP long olayı vardı.
Giriş/çıkışları karar sonrasında gerçekten gözlenmiş ask/bid fiyatlarıdır;
gerçek emir veya fill değildir. İlk iki olayın ortalaması:

| Ölçü | Baz puan |
|---|---:|
| Mid-price hareketi | −0,1284 |
| Gözlenen spread etkisi | 0,0129 |
| Ask'tan giriş/bid'den çıkış brüt hareketi | −0,1414 |
| Varsayılan ücret ve kayma etkisi | 23,9709 |
| Model net hareketi | **−24,1122** |

İki olayın model neti yaklaşık **−%0,2802** ve **−%0,2021**.
Sabit 1.000 USDT olay büyüklüğünde toplam hipotetik fark **−4,82 USDT**.
Bu, gerçek hesap zararı veya portföy getirisi değildir. Kontrol iki olayda da
flat kaldı; sıfır tamamlanmış kontrol işlemi eksik fiyat veya başarılı işlem
anlamına gelmiyor. Flat sayısı ayrıca raporlanıyor.

Bu iki gözlem bir stratejiyi istatistiksel olarak kabul veya reddetmeye yetmez.
Fakat maliyet sorunu görünür: bu örnekte brüt hareket birkaç baz puanken ücret
ve kayma varsayımı yaklaşık 24 baz puan. Fiyat hareketi pozitif olan ikinci
olay da maliyet sonrası ekside. Spread ile varsayılan ücret/kayma ayrı
hesaplanıyor; aynı maliyet iki kez sayılmıyor.

O andaki son uygun 15 dakikalık pencerede yaklaşık **729 dolar** signed BTC
TWAP aktivitesi, **1 hesap** ve **1 parent-ID** vardı. En büyük hesap payı
%100, etkin hesap sayısı 1. Bu hesap çıkarılınca net akış sıfır. Bu değerler
tek hesabın küçük programlı alımını gösterir; geniş piyasa uzlaşması değildir.
Parent-ID, kalan emir büyüklüğünü veya iptal durumunu açıklamaz.

[İlk iki olayın analiz kaydı](evidence/alpha-review-20261031/initial-economics.json)
tarihli tutuldu. Daha sonraki tutarlı checkpoint'in zamanı ve güncel olay
sayısı [manifestte](evidence/alpha-review-20261031/manifest.json), sonuçları
[kayıtlı raporda](evidence/alpha-review-20261031/forward/recorded-report.json)
ve [ekonomik analizde](evidence/alpha-review-20261031/forward/diagnostics.json).

**11:48 UTC checkpoint'inde** üçüncü olay da tamamlandı: toplam 3 tamamlanmış
TWAP olayı, ortalama model neti **−17,0788 bp** (−%0,1708), toplam hipotetik
fark **−5,12 USDT**. Belirsiz aktif sonuç 0; sonraki eşleştirilmiş olay hâlâ
bekliyor. Üç sonuç da maliyet sonrası ekside. Bu sayı hâlâ genelleme için çok
küçük; kayıtlı rapor ilk iki olayın tarihli analizinin yerine sessizce geçmez.

## Yeni aktif TWAP kaynak taraması

Tarama **11:37:21 UTC**'de, ilk ağ çağrısından önce kaydedilmiş
[sözleşmeyle](evidence/alpha-review-20261031/discovery/discovery-spec.json)
başladı; **11:39:03 UTC**'de tamamlandı. Kârlı hesap veya sonradan bilinen PnL
etiketi kullanılmadı. BTC recentTrades katılımcılarından 30 sınırlı snapshot
alındı; mevcut 22 hesap dışarıda bırakıldı. En çok 96 yeni adres ve eşzamanlı
iki salt okunur istek sınırı vardı.

Önceden belirlenen kaynak uygunluğu: ortak son 15 dakikalık pencere içinde
mutlak net BTC TWAP notional'ı **en az 100.000 dolar**, son dilim **en çok
120 saniye** eski; limitli yanıt pencere başına uzanmalı. Bu büyüklük eşiği
ekonomik kaynak ön elemesidir; tahmin veya kârlılık kanıtı değildir.

- 86 benzersiz public katılımcı görüldü; 5'i mevcut örneklemdeydi.
- **81 yeni hesabın 81'i** okunabildi; istek hatası yok.
- 17 hesapta yakın tarihli BTC TWAP dilimi vardı, 64'ünde yoktu.
- En büyük mutlak net akış yaklaşık **11.385 dolar**; uygun hesap **0**.
- 111 ham yanıt arşivinin hash'i ve seçim hesabı yeniden doğrulandı.

[Sonuç](evidence/alpha-review-20261031/discovery/summary.json), negatif tarama
sonucudur. Bütün Hyperliquid'de büyük TWAP olmadığı anlamına gelmez; yalnız
bu kısa aktivite örneklemi tarandı. Eşik sonradan düşürülmedi ve yeni cohort
oluşturulmadı. Native recent-list kapsamı, bütün borsa geçmişi değildir.

## Fonlama maliyeti ön elemesi

[Ayrı sözleşme](evidence/alpha-review-20261031/funding/probe-spec.json), BTC,
ETH, SOL ve HYPE için geçmiş **168 saatlik** native Hyperliquid fundingHistory
okumasından önce donduruldu. Her varlıkta 168/168 beklenen saatlik ödeme
mevcut; eksik saat, geleceğe yazılmış kayıt veya çift sayım bulunmadı.

Varsayım: spot long ve eşit sabit perp notional'ı short. Pozitif fonlamayı
short alır; negatif fonlamayı öder. İki yönde spot ücreti 8 bp, perp ücreti
4,5 bp; dört işlem bacağı için 4'er bp kayma: toplam **41 bp**.

| Varlık | Geçmiş 7 günlük fonlama, bp | Fonlama − 41 bp varsayılan maliyet |
|---|---:|---:|
| BTC | 19,32 | −21,68 |
| ETH | 8,75 | −32,25 |
| SOL | 17,60 | −23,40 |
| HYPE | 35,52 | −5,48 |

Bu tablo net strateji getirisi değildir. Uygulanabilir spot/perp spread ve
basis değişimi, spot varlığın ilgili venue'de bulunması, teminat/finansman
maliyeti, stablecoin dönüşümü ve transferler doğrulanmadı. Toplam sermayeye
getiri veya APY hesaplanmadı. Saatlik geçmiş oran basit yıllıklandırılırken
24×365 kullanılır; oran geleceğe sabitlenmez. Fonlama yön değiştirebilir.

[Sonuçlar](evidence/alpha-review-20261031/funding/summary.json) yalnız bu
varsayımlarla, yedi günlük fonlamanın varsayılan işlem maliyetini
karşılamadığını gösterir. Gerçek kullanıcı fee tier'ı farklı olabilir;
bu çalışma bütün carry yollarını reddetmez. Dört ham yanıt arşivi ve hesaplar
çevrimdışı yeniden doğrulandı. Hiçbir sonuç emir açmaz.

## Veri toplamanın devamı

Likidasyon + spot absorpsiyonu çalışması için önceki public kaynak worker'ları
**31 Ekim 11:43 UTC** civarında aynı donmuş hesaplarla yeniden başlatıldı:
OKX 30 saniye, ilk altı hesap 120 saniye, ilave 16 hesap 240 saniye aralıkla.
TWAP forward worker'ı kesilmeden devam etti. Dördünün duruş bütçesi
**1 Kasım 09:00 UTC**; bu tarihte Londra saatiyle aynı. Ortam kapanırsa
süreçlerin yaşamaya devam edeceği varsayılmaz. Her worker düşük disk alanında
durur. Başka kullanıcılara bildirim, ücretli API veya emir yok.

[Başlangıç planı](evidence/alpha-review-20261031/collection-resume-plan.json)
ağ çağrısından önce eski journal uç kayıtlarının hash'lerini ve satır
sayılarını sakladı. İlk yeni döngülerden sonra bu kayıtların korunduğu
doğrulandı. Gündeki kaynak boşluğu ilk yeni health kayıtlarında açıkça
işaretlendi; envanter değişimi başlangıçta `missing_reference`, OKX kapsamı
başlangıçta belirsiz. Bunlar sıfır akış veya sağlıklı geçmiş sayılmıyor.

Yeniden başlama sonrası [salt okunur tam denetimde](evidence/alpha-review-20261031/resumed-data-audit.json)
üç journal'daki **126.875** payload hash'i eşleşti; zaman ve OKX tutar
kontrolleri geçti. Yeni forward checkpoint'indeki **1.036** ham yanıt/fiyat
kaydı, giriş/çıkış quote referansları ve net model hesabı da doğrulandı.
Bunlar kaydedilmiş verinin iç tutarlılığıdır; tam upstream kapsam onayı değildir.

Bu kaynak toplaması yeni, donmuş likidasyon stratejisi testi değildir.
Önceki likidasyon replay'i de açıklayıcı analiz olarak kalır. Sonraki bağımsız
test; açıkça gerçekleşmiş likidasyon, gözlenmiş spot akışı, zamanında fiyat
referansı ve maliyet kapısını birlikte gerektirecek. Yeni veri, bu kararları
geçmişe yazmak için kullanılmayacak.

## Teslim ve yeniden üretim

Arayüz brüt mid hareketi, spread etkisi, ücret/kayma etkisi, model neti,
kontrolün flat sayısı ve hesap yoğunlaşmasını gösteriyor. Analiz
`HYDRA_FORWARD_TRIAL` ile seçilen günlüğü salt okunur transaction içinde okur;
sayfa açmak upstream veri çağrısı veya sonuçlandırma yapmaz. Eski kayıtlara
taze zaman damgası verilmez. 221 uygulama testi geçti; sentetik fixture'lar
alfa sonucu değildir. Chromium masaüstü ve 390 px mobil kontrolleri geçti.

İlk iki teslimin protokolü ve hash'li arşivleri korunmuştur. Yeni checkpoint
canlı SQLite'tan backup API ile alındı; açılmış dosyadan aynı rapor ve maliyet
analizi yeniden üretilebilir. [Runbook](RUNBOOK_TR.md) komutları, snapshot'ın
`protocol.json` ve açılmış `ledger.sqlite3` dosyalarının bulunduğu dizine
uygulanır. Kaynak taramaları `--audit --out <arşiv-dizini>` ile tekrar hesaplanır.

Kod `research/twap-forward-20261031` dalında teslim edilir. Yerel/kayıtlı
önizleme Railway yayını değildir. Emir ve otomatik strateji terfisi kapalıdır.
