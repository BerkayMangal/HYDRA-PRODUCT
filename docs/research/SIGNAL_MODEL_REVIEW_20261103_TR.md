# HYDRA canlı sinyal modelleri incelemesi — 3 Kasım 2026

**Esas bulgu:** canlı arayüzdeki BTC/ETH modelinin `/api/scalp` özetinde 156
sonuç, 99 kazanç (%63,5) ve işlem başına **−%0,083 ortalama getiri** var.
Bu modelin yüksek isabeti, mevcut toplam özette pozitif getiri anlamına gelmiyor.
Son dönemde 4 saatlik model daha iyi görünüyor; maliyetler bu farkı silebiliyor.

İnceleme: 15:54–16:04 UTC civarında canlı Railway uygulamasının salt okunur
API cevapları. Kaynak `https://web-production-57fc0.up.railway.app/`.
Ham API cevapları ve yeniden hesaplanan sayılar `evidence/signal-review-20261103/`
altında. Bu inceleme model ayarını değiştirmedi, mesaj göndermedi ve işlem yapmadı.

## Hangi model?

Canlı ana sayfadaki **“BTC & ETH signal — 15-minute models”** bölümü
`/api/scalp` kullanıyor. Bu endpoint ve `config/models/*.joblib` dosyaları bu
checkout'ta yok. Çalışan model kaynak koduna, tam işlem defterine ve eğitim
verisine bu repo üzerinden erişilemedi. Ana sayfanın `/assets/hydra.js` dosyası
ve sunduğu ekran da bu checkout'ın `static/index.html` dosyasından farklı.
Çalışan Railway kaynağı ile bu GitHub main sürümünün eşleştiği doğrulanmadı.
Bu nedenle eski uygulamayı canlı sitenin üzerine yayınlayarak model değiştirilmedi.

Aşağıdaki rakamlar **canlı API'nin bildirdiği simüle edilmiş sinyal sonuçları**;
borsa gerçekleşme kayıtları, müşterinin hesabı veya bağımsız tekrarlanmış backtest
sonucu değil. `validated: true` etiketi tek başına bağımsız doğrulama sayılmadı.

## Bildirilen geçmiş test

| Model | İşlem | İsabet | Ort. işlem getirisi | Sharpe | Pozitif dönem |
| --- | ---: | ---: | ---: | ---: | --- |
| 15dk ensemble, en fazla 30dk | 349 | %62,2 | %0,0360 | 1,00 | 14/21 ay |
| 15dk HGB, en fazla 4s | 2.051 | %66,9 | %0,0356 | 1,83 | 15/21 ay |
| Swing XGB, %3 stop / %5 hedef | 306 | %41,2 | %0,1450 | 0,45 | 4/7 çeyrek |

API'ye göre seçim dönemi 2022-01-01 → 2025-01-01, OOS başlangıcı 2025-01-01.
Fakat çalışma **2026-09-26** tarihinde yayımlanmış; arayüzdeki “→ today” ifadesinin
3 Kasım'a kadar yeniden hesaplandığı bu cevapla doğrulanmıyor. `data_source` null.
Bu tablonun ortalama getirilerinde hangi komisyon/kaymanın düşüldüğü belirtilmiyor.

4 saatlik modelin bildirilen ortalama avantajı **3,56 baz puan/işlem**. Eğer bu
rakam brütse örneğin 10 baz puan toplam işlem maliyetini karşılamıyor. Zaten netse
aynı maliyet ikinci kez düşülmemeli; kaynak ve tam defter olmadan ayrım doğrulanamadı.
API'nin `total_pct` rakamlarını bileşik, eşzamanlı pozisyonları hesaba katan portföy
getirisi olarak kullanmadık. `model_checks` içindeki 48 satırlık `max_diff=0`
kontrolleri de kârlılığın veya bağımsız testin kanıtı değil.

## Canlı kayıt ve son işlemler

`record`: n=156, wins=99, win_rate=0,635, avg_pct=−0,083.
Bu özetin 156 satırlık ham defteri endpoint'te yok. `recent` yalnızca 20 satır
veriyor; bunların ikisi hâlâ açık. Tam örneklemin model bazlı katkısını, düşüşünü
ve gerçek portföy getirisini bu kısıtlı listeden yeniden üretmek mümkün değil.

20 satırlık listede, 27 Ekim → 3 Kasım tarihleri arasında:

| Alt örneklem | Kapanış | Kazanan | İsabet | Ort. bildirilen getiri |
| --- | ---: | ---: | ---: | ---: |
| Tüm kapanışlar | 18 | 14 | %77,8 | −%0,0591 |
| `sent=true` kapanışlar | 14 | 10 | %71,4 | −%0,1701 |
| 4s HGB, gönderilmiş ve gönderilmemiş | 17 | 14 | %82,4 | +%0,1139 |
| 4s HGB, yalnız `sent=true` | 13 | 10 | %76,9 | +%0,0476 |

Diğer bir kapanış swing ETH'de −%3. Örneklem küçük ve seçilen son pencere; bu
karşılaştırma swing modelinin tamamının zararlı olduğunu göstermiyor. Öte yandan
4 saatlik modele ait son listede gönderilmemiş dört işlemin dördü de kazanmış.
Bu yüzden yalnız iyi görünen tüm sinyalleri, müşteriye iletilenlerle aynı sonuç
olarak sunmamak gerekiyor. `record` ile `sent` filtresinin kaynak kodundaki tam
ilişkisi doğrulanamadı.

Son kapanışların `ret_pct` alanı, `(exit_price / entry - 1) × 100` ile yuvarlama
hassasiyetinde aynı: en büyük fark %0,000478'den küçük. Bunlar görünürde **brüt
fiyat hareketleri**; bu satırlarda ücret ve kayma alanı yok. Örnek olarak 3 Kasım
11:15 ETH alımı 2.269,65 → 2.275,5903, bildirilen +%0,262. Aynı gün 13:45 ETH
alımı 2.293,50 → 2.277,4974, bildirilen −%0,698. Bir kayıp birkaç kazancı silebiliyor.

## Maliyet duyarlılığı — senaryo, gerçek gerçekleşme değil

Aşağıdaki tutarlar gidiş-dönüş toplam maliyet varsayımları. Müşterinin gerçek
ücretleri, emir türü ve kayması ayrıca ölçülmeli. Ortalama eşit ağırlıklı işlem
getirisi; portföy getirisi değil. Açık işlemler dahil edilmedi.

| Son pencere | Brüt ortalama | Toplam 10 bp maliyetle | Toplam 14 bp maliyetle |
| --- | ---: | ---: | ---: |
| 17 HGB kapanışı | +%0,1139 | +%0,0139 | −%0,0261 |
| 13 gönderilmiş HGB kapanışı | +%0,0476 | −%0,0524 | −%0,0924 |

Bu yüzden builder ücreti de diğer maliyetlere eklenerek değerlendirilmeden
“HYDRA'da kazanıyor, üstüne komisyon alırız” sonucu çıkarılmamalı. Ek ücret mevcut
ince avantajı tüketebilir. Bu araştırma için gerçek emir veya ücret onayı gönderilmedi.

## Diğer motorlar aynı performans defteri değil

- Eski `/api/performance` Layer 1 tracker'ı: 0 sinyal, 0 bekleyen, boş liste.
  **Bu sıfır, yukarıdaki BTC/ETH modelinin kayıtsız olduğunu göstermiyor.**
- Eski `/api/ml` XGBoost engine: konfigürasyonda kapalı. Bu da `/api/scalp`
  içindeki çalışan üç modelin kapalı olduğu anlamına gelmiyor.
- AI Pulse: GPT-4o-mini, BULLISH ve 65 güven; agreement=SINGLE. 65, doğrulanmış
  %65 kazanma olasılığı değil. İlgili canlı cevap tarihli sonuç defteri sağlamıyor.
- `/api/alpha/performance`: ayrı motor; 5.000 açık, 0 çözülen kayıt ve kalibrasyon
  n=0. 5.000 sayısı kapanmış başarılı işlem sayısı değil.
- Discovery scorecard: 116 tip/ufuk satırında 70 collecting, 29 failed,
  17 inconclusive; proven satır yok. Ufuklar aynı olayları tekrar kullanabildiği
  için bu satır sayıları bağımsız deney sayısı olarak toplanmamalı.
- VENOM'un beş farklı coin kartında aynı 24s `forward_edge` özeti görünüyor
  (n=376, ortalama −%1,052, win_rate=null). Bu metrik coin bazında doğrulanmış
  ayrı getiri gibi okunmamalı; ortak özet mi yanlış eşleme mi, canlı kaynak
  olmadan belirlenemedi.

## Sayıları yeniden hesaplama

Repo kökünde, ağ bağlantısı ve cüzdan olmadan:

```sh
python -m research.lab.signal_snapshot_review docs/research/evidence/signal-review-20261103/scalp.json
```

Çıktıdaki dosya SHA-256 değeri hangi anlık kaydın incelendiğini sabitler.
Maliyet senaryoları son kapanışların brüt fiyat getirisine uygulanır; OOS tablosuna
net/brüt belirsizken tekrar maliyet düşülmez.

## İzlenecek sıra

Önce çalışan `/api/scalp` sürümünün kaynak ve model dosyalarını bu repo ile
uzlaştırmak. Sonra 156 sonucun tam defterini alıp model, BTC/ETH, gerçekten
iletilen sinyal ve zaman dönemi bazında; ücret, kayma ve eşzamanlı pozisyonlarla
tekrar hesaplamak. 4s HGB şu an incelenmeye en değer aday; bu sonuç onu otomatik
canlı işlem stratejisi yapmaya yetmiyor. Modeli değiştirmeden sonraki ileri dönem
ölçümü, bağımsızlığı korur. Spot cüzdan ekranı bu modele otomatik bağlanmadı.

## Ek: repodaki eski Layer 1 / ML kodunun sınırları

Bu kısım **yalnız bu checkout'taki eski modüller** için kod incelemesi. Çalışan
`/api/scalp` modeline uygulanmış kusurlar olarak yorumlanmamalı.

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
