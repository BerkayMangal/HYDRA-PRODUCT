> Tarihli araştırma/audit kaydıdır. Güncel kod, bulgular ve çalışma durumu için [29 Ekim teslimini](FINAL_TR.md) ve [çalıştırma kılavuzunu](RUNBOOK_TR.md) okuyun. Bu kayıttaki eski hata ve süreç durumları bugünkü durumun beyanı değildir.

# 28 Ekim 2026 — veri bütünlüğü ve EMA/spot geliştirme paketi

Bu çalışma ilk kod düzeltme paketidir. Yerel checkout değiştirildi; commit/push,
Railway yayını, gerçek emir gönderimi ve bildirim yapılmadı.

## Tamamlanan uygulama düzeltmeleri

- Collector hata verdiğinde eski cache'i yeni veri diye döndürmüyor; boş
  yanıtlar da hata sayacına giriyor. Orkestratör önceki veriyi kendi özgün
  zamanıyla tutuyor; daha eski bir gözlem yenisini ezemiyor.
- Hesaplamalar ortak ham veriyi değiştirmiyor. Fiyat, stablecoin market cap
  veya başka bir gerçek değer TTL nedeniyle sıfıra doğru küçültülmüyor.
- Z-score geçmişleri her özellik için ayrı kaynak zamanı kullanıyor. Aynı
  gözlem tekrar okununca örnek sayısı artmıyor; süresi geçen değer eski son
  geçerli sayıdan yeniden üretilemiyor.
- Collector özellikleri her frame rebuild'de tazelenmiyor. Kaynak zamanı
  kaliteye taşınıyor; provenance bulunmayan veya geleceğe tarihli değer missing.
  Legacy engine girişinde missing özellikler dışlanıyor.
- Kritik BTC fiyatı ve kapanmış mum eksik/eskiyse kalite kapısı RED oluyor.
  RED veya hesaplama hatası mevcut dashboard/hybrid yönünü NO_SIGNAL'a çeviriyor.
- Kalite çarpanı yön/state seçilmeden önce uygulanıyor. Düşük kalite, eşiği
  geçemeyen puanın önceki güçlü state'i korumasına neden olmuyor.
- OKX ticker freshness'i borsa event zamanını kullanıyor. CCXT senkron
  istekleri ortamın HTTPS proxy'sini kullanıyor; TLS doğrulaması açık.
- Teknik göstergeler için borsanın kapanmış 1H mumları alınıyor. Fallback 5m
  akışında yinelenen/sırasız mumlar sayılmıyor; 12 farklı mum olmadan saat
  tamamlanmıyor. Kapanmamış mumlar ve hatalı OHLC reddediliyor. Saat boşluğu
  varsa aradaki eksik saatleri yok sayarak getiri hesaplanmıyor.
- Macro cache'in eski değerleri yeniden çekilmiş gibi tazelenmiyor.
  Tarihsiz intraday örneklerden sahte 5 günlük momentum/altın oranı çıkarılmıyor.
- ETF parser tarihli Total sütununu ve çok satırlı vendor başlığını okuyor;
  parantezli negatifler, binlik ayraçlar, sıfır ve eksik ayrılıyor. Fonlar toplamı
  Total ile uyuşmalı. Pending satır sıfır akış sayılmıyor. `7d` artık son akış
  tarihinde biten yedi takvim günü; gün sayısı ve pending tarihi açık.
- Geniş stablecoin/kripto market cap proxy'si farklı bir isimle saklanıyor;
  CryptoQuant exchange ratio alanının üzerine yazılmıyor.
- Liquidation endpoint'i doğru instFamily ile çağrılıyor; enstrüman, gerçek
  60 dakika penceresi ve kontrat çarpanı kullanılıyor. Public veri örneklem
  olduğu için `liq_sample_only` ile işaretleniyor.
- Dashboard gerçek float engine şemasını kabul ediyor; history güncellemesi
  exception üretmiyor. Gerekli HTTP yanıtları başarısızsa arayüz Live yazmıyor.

## Sabit EMA + spot karşılaştırması

Kaynak: `HYDRA-PRODUCT/research/spot_confirmation.py` ve
`research/spot_confirmation_spec.json`. Baseline önceki araştırmanın
`806ef45dc42abb7a` adayı: kapanmış 4H BTC/USDT spot, EMA4 > EMA200, long/flat.
Parametreleri değiştirilmedi. Yeni doğrulama, 5m ve 15m spot quote-volume
alış/satış dengesinin ikisinin de pozitif olmasını ister.

Kaynak event zamanı ve gerçek receipt zamanı decision timestamp'inden sonra
olamaz. Spot trade ID örtüşmesi, poll boşluğu, işlem güncelliği ve mum sürekliliği
kontrol edilir. Eksik pencere nötr/pozitifmiş gibi doldurulmaz. Baseline dışında
pozisyon oluşturulmaz. OI/funding bu ilk karşılaştırmada kayıt/context olarak
kalır; bu değişiklikte karar filtresine eklenmedi.

İleriye dönük iki varyant aynı karar anı ve fiyat referansıyla saklanıyor.
Sanal envanter muhasebesi, her forecast'te yeniden komisyon kesmek yerine
yalnız pozisyon değişiminde maliyet uygular. Veri eksikse filtreli araştırma
portföyü mevcut kaydedilmiş referans fiyatıyla nakde döner. Bu, broker fill'i
veya gerçekleşmiş para getirisi değildir. Sıfır/az işlem alfa doğrulamaz.

Gölge worker sadece bu cloud makinesinde yeniden başlatıldı; hedef duruş
**29 Ekim 2026 08:00 Türkiye saati**. Makinenin çalışması şarttır. İlk başlatma
boşluğu kayıtlarda korunur; yeni pencerenin ısınması beklenir. Saatlik karar
zamanından önce filtre başarısı raporlanmaz. Güncel karşılaştırma raporu:
`SPOT_CONFIRMATION_TR.md`; SQLite tablo adı: `comparison_decisions`.

## Doğrulama

- Uygulama suite: **79 passed**, tek mevcut pandas test uyarısı.
- Yardımcı collector suite: **18 passed**.
- Nedensellik, kaynak zamanı, duplicate/gap/TTL, ETF tablosu, RED state,
  kalite çarpanı, kontrat notional, farklı proxy davranışı ve sanal maliyet
  muhasebesi için regresyon kontrolleri eklendi.
- Eski test fixture'ının high/open/low sınır ihlalleri düzeltildi; gerçek 1H
  candle giriş yolu kullanıldı. Test assertion'ları zayıflatılmadı.
- Sabit EMA sonucu 2021–2026 yıllarının altı kontrol noktasında eski araştırma
  adayıyla aynı çıktı. Donmuş kuralın SHA-256 değeri doğrulandı.
- Gerçek OKX ile yerel uygulama bir tam karar döngüsünü tamamladı. Ana sayfa,
  health/status/signal/decision/docs uçları 200; signal ve decision state/puanı
  aynı; market fiyatı mevcut, data-store durumu dolu. Bildirim anahtarları test
  child process'inde kaldırılarak dışarı mesaj gönderimi engellendi.
- Arşivlenmiş gerçek Farside tablosu yeni parser ile kontrol edildi:
  26 Ekim günlük +$110,3M; son yedi takvim günü -$115,2M; 27 Ekim pending.

## Kalan sınırlar

Bybit bu makineden bölgesel erişim kısıtıyla HTTP 403 veriyor; anahtar eklemekle
çözüldüğü varsayılmadı. Eksik özellikler gizlenmiyor. Canlı Railway sürümü
checkout ile eşleştirilmiş değil; bu yerel testler deploy kanıtı değildir.

Eski ML walk-forward simülatöründeki zaman eşleme, label/imputation ve OOS
çifte sayım sorunları bu paketin kapsamına alınmadı; ML kapalı kalmalı.
Legacy CVD'nin sabit trade sayısı/pencere anlamı ve macro US10Y değişim birimi
ayrı düzeltme gerektiriyor. Yeni EMA/spot araştırması bunları karar girdisi
olarak kullanmıyor. Bu paket tüm platformun veri doğruluğuna veya kârlılığına
genel onay anlamına gelmez.

## Dosyalar

- İncelenebilir tüm kaynak değişiklikleri: `changes-20261028.patch`.
- Uygulama test çıktısı: `application-tests-20261028.log`.
- Gerçek uygulama kontrolü: `functional-result-20261028.json`.
- Gölge worker: `shadow.py`, `shadow.log`, `shadow-status.json`.
- Kurallar/sonuçlar: repo `research/` paketi, `comparison-summary.json`,
  `SPOT_CONFIRMATION_TR.md` ve SQLite journal.

Değişiklikler çalışma dizininde incelemeye hazırdır; patch mevcut kirli
checkout'a yeniden uygulanmamalıdır.
