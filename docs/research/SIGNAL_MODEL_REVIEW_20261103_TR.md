# HYDRA canlı sinyal modeli incelemesi — 3 Kasım 2026

İnceleme: canlı Railway uygulamasının 15:54–15:55 UTC civarındaki salt okunur API
cevapları ve bu çalışma dalındaki kaynak kod. Canlı uygulamanın sürüm endpoint'i
olmadığından kaynak kod ile çalışan sürümün birebir eşleştiği doğrulanamadı.
Ham cevaplar `evidence/signal-review-20261103/` altında. Bu inceleme mevcut sinyal
modelini değiştirmedi, ML eğitimini başlatmadı ve işlem yapmadı.

## Ölçülebilen sonuç

| Gösterge | Canlı gözlem |
| --- | --- |
| Çalışan sinyal | Layer 1; mikro yapı, akış ve makro bileşimi |
| O andaki yön | NEUTRAL; skor yaklaşık −0,83 |
| Arayüzde güven | HIGH; kalibre edilmiş kazanma olasılığı değil |
| Girdi tamlığı | 0,91; bu da başarı oranı veya doğruluk ölçüsü değil |
| Performans örneklemi | 0 kayıtlı sinyal, 0 bekleyen, boş son sinyal listesi |
| 1s / 4s / 24s başarı | Ölçülemiyor; API'nin sıfırları örneklem yokluğundan |
| Ayrı XGBoost ML modeli | Konfigürasyonda kapalı; eğitilmiş değil, 0 işlem |
| Servis oturumu | Yaklaşık 1 gün 16 saat; 1.362 döngü |

**Sonuç:** sistem güncel görüş üretiyor, ancak erişilebilen kayıtlarla kârlı
çalıştığını veya geçmişteki başarı oranını söylemek mümkün değil. Sıfır kayıt,
modelin başarısız olduğunun kanıtı da değil. Bu oturumdan önceki geçmişe erişim
sağlanmadı. O andaki nötr görüşün sonradan doğru çıktığı da bu tek anlık veriyle
ölçülemez.

## Kayıt ve ölçüm sınırlamaları

- `main.py`: yalnızca olgunluk kontrolünü geçen BULLISH/BEARISH kararları tracker'a
  giriyor. NEUTRAL ve WEAK kararları kayıtlı sonuç örneklemini oluşturmuyor.
  Telegram gönderimi ile tracker kaydı aynı hata bloğunda; gönderim istisnası
  kaydı da engelleyebilir. Boş canlı kayıtların kesin nedeni geçmiş günlükler
  olmadan belirlenemedi.
- `services/signal_tracker.py`: tracker bellekte tutuluyor; bu sınıfta kalıcı
  defter yükleme/kaydetme yok. En fazla 500 tamamlanmış kayıt korunuyor.
  Dolayısıyla yeniden başlatmalardan önceki ömür boyu performansı temsil etmez.
- HIT/MISS, yönün lehine fiyat hareketi olmasına bakıyor. Ortalama getiri
  komisyon, spread ve kayma düşülmüş portföy kârı değil. SHORT sonuçları da spotta
  BTC satın alan bir müşterinin kazancı olarak kullanılamaz.
- 1/4/24 saat ufuklarında sonuç, hedef süre geçtikten sonraki ilk kontrolün
  fiyatıyla ölçülüyor; tolerans 1/4 saat için bir saat, 24 saat için iki saat.
  Tam hedef zamandaki fiyatı garanti etmiyor. Süresi geçen kayıtlar EXPIRED
  oluyor ve başarı paydasından çıkıyor; kapsama da ayrıca raporlanmalı.
- Birbirine yakın sinyallerin 24 saatlik getirileri örtüşebilir. Bu sonuçların
  standart sapmasından `sqrt(365)` ile türetilen Sharpe proxy, bağımsız günlük
  portföy getirilerinin Sharpe oranı olarak sunulmamalı.

## Ek veri ve ML bulguları

Canlı status cevabında makro dahil bazı kaynaklar yaklaşık 53 dakika yaşındaydı
ve `is_stale=true` idi. Aynı cevaptaki `bybit_mark_price` yaklaşık `3,05e-19`,
ana BTC fiyatı yaklaşık 77.600 idi. Bu alanlar tutarsız; mevcut motorun bu Bybit
alanını karar için kullandığı bu incelemede gösterilmedi. HIGH/0,91 etiketleri
tek başına tüm kaynakların taze veya doğru olduğunu kanıtlamıyor.

Ayrı ML modeli açılmadan önce incelenmeli:

- Eğitim kodu son %30'luk aynı kronolojik bölümde kalibrasyon, metrik ve eşik
  seçimi yapıyor. Bu bölüm bağımsız, hiç kullanılmamış nihai test seti değil.
- Paper-trade kodundaki USDC için %5 yıllık varsayımsal getiri, geçen zamana göre
  değil `_update_paper_trade` çağrı sayısına göre ekleniyor. Saatlikten sık
  çağrılırsa getiri abartılır. Canlı ML kapalı olduğu için bu bulgu mevcut bir
  kazanç rakamını doğrulamıyor veya açıklamıyor.
- Paper-trade `win_rate`, kapanışın brüt `pnl_pct > 0` olmasına bakıyor; tüm
  maliyetler sonrası kazanan işlem oranıyla aynı değil.

## Önerilen ölçüm

Modeli değiştirmeden önce her karar için kalıcı UTC kayıt, sürüm, kaynak yaşı,
karar fiyatı ve nedenini saklamak; 1/4/24 saat sonuçlarını ayrı zaman damgalı fiyatla
sonradan eklemek. Yön isabetinden ayrı olarak spotta uygulanabilir BUY/HOLD/USDC
portföyünü, gerçekçi ücret/kaymayla ve al-tut karşılaştırmasıyla ölçmek. Eksik ve
süresi geçen gözlemleri de göstermek; aynı günün örtüşen kararlarını bağımsız
başarılar gibi saymamak. Mevcut API bunun geçmiş verisini sağlamadığı için bu
raporda uydurma backtest veya kâr hesaplanmadı.
