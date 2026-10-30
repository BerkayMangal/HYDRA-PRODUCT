# 30 Ekim — zaman penceresi, maliyet ve native TWAP teslimi

Sürüm `hydra-research-20261030.1`. Önceki uygulamanın üzerine yapılan bu çalışma
gerçek para emri, yeni PnL sıralaması veya strateji terfisi içermiyor.

## Zamanlama hatası ve gerçek veri kontrolü

16 hesaplık toplayıcıda 240 saniye bekleme API çevriminden sonra başlıyordu.
Yaklaşık 32 saniyelik çevrimle gözlem aralığı 4,53 dakikaya çıkıyor, 15 dakika
farkı için gereken 14–18 dakika aralığı çoğu zaman boş kalıyordu. 289 eski
snapshot'ın 247'sinde fark ölçülememişti.

Yeni zamanlama çevrim başlangıcını kullanır, kaçan slotlarda telafi istekleri
yağdırmaz. Donmuş 6+16 hesap ve 14–18 dakika sözleşmesi korunur. Eski veriler
geriye dönük tamamlanmaz. Snapshot/health kaydı bekleme, eksik referans ve eksik
hesabı ayrı gösterir. Dashboard ve rapor da mevcut/eksik pencere sayısını sunar.

Kontrol ölçütleri yeni okumadan önce 07:05 Londra'da kaydedildi. 07:22 kontrolü:

| Örneklem | Yeni snapshot | Başarılı hesap okumaları | Geçerli fark penceresi | Medyan gözlem aralığı |
|---|---:|---|---:|---:|
| İlk 6 | 9 | Her snapshot 6/6 | 2 | 119,5 saniye |
| Ek 16 | 5 | Her snapshot 16/16 | 1 | 239,8 saniye |

Ek örneklemin ilk geçerli farkı 15,9608 dakika. Önceki bütün feature satırlarının
zaman/payload SHA-256 dizisi aynı kaldı. Bu kısa kontrol zamanlama düzeltmesini
destekler; her gelecekteki API çağrısının eksiksiz olacağı veya alfa bulunduğu
anlamına gelmez. Başlangıçtaki eksik pencereler de kayıtta tutuldu.

Ölçüt ve sonuçlar [kanıt dizinindedir](evidence/cadence-20261030/).

## Kullanıcıya göre maliyet senaryosu

Cost Scenarios alanında komisyon ve fiyat kayması taraf başına, funding ise
ufuk başına varsayımsal bir ödenek olarak girilir. Varsayılan 8+4 bp taraf
maliyeti, 24 bp gidiş-geliş eşik oluşturur. 1 bp = %0,01. Kişisel gerçek
komisyon/funding ölçülmüş gibi sunulmaz.

Kayıtlı ilk örneklemin ortalama yönlü fiyat hareketinden bu varsayımlar çıkarılır.
Gözlem yoksa sonuç bilinmiyor kalır. Ufukların örtüşmesi ve yoğunlaşmış örneklem
nedeniyle bu tablo gerçekleşmiş net getiri veya portföy backtest'i değildir.
Pozitif fark alfa onayı üretmez. Kullanıcı girdisi sadece kendi isteğine etki
eder; uygulama ayarını, cüzdan seçimini ve karar motorunu değiştirmez.

API: `/api/research/costs`. Mevcut tarihsel taramada kabul ölçütlerini geçen
strateji sayısı hâlâ sıfır; ML ve emir gönderimi kapalı.

## Ücretli sağlayıcı olmadan native TWAP denemesi

Hyperliquid'in resmî Python SDK kaynağındaki `userTwapSliceFills` public `/info`
okuması doğrulandı. SDK kurulmadı veya çalıştırılmadı. Native query whitelist'ine
yalnız bu bilgi türü eklendi; emir/imzalama türleri reddedilmeye devam ediyor.

07:13 Londra'da **22/22 hesap** başarıyla okundu. İki hesapta sırasıyla 1.510 ve
1.047 tekil BTC slice fill'i bulundu. İkisi de 2.000 toplam satır sınırına
ulaşmıştı; bu sayılar tam tarih veya bütün piyasa kapsamı değildir. Son BTC
olay zamanları 29 Ekim 23:54 ve 30 Ekim 07:13 UTC idi. Geçmiş fill'ler receipt
anında yeni gerçekleşmiş işlem diye etiketlenmedi.

Endpoint parent programın kalan miktarını veya iptal durumunu vermiyor.
Bu deneme TWAP veri yolunun kullanılabilirliğini gösterir; devam eden program,
gelecek alım/satım veya kârlı strateji kanıtlamaz. Sonuç dashboard Research
Evidence alanında tarihli deneme olarak görünür.

Ham 22 yanıt gzip ve SHA-256 ile [kanıt dizininde](evidence/twap-20261030/)
korundu; hash'ler ayrıca doğrulandı. Yeni bağımsız deneme:

```bash
python -m research.lab.twap_probe
```

## Doğrulama ve yayın

179 test geçti; iki mevcut deprecation uyarısı var. Docker ve gerçek Chromium
kontrolünde 22/22 canlı wallet okuması, 19 kaynak bağlantısı, maliyet formunda
24→28 bp güncellemesi ve mobil 390 px görünüm kontrol edildi; JavaScript
exception görülmedi. Güncel deploy kontrolleri [dağıtım kılavuzundadır](../DEPLOYMENT_TR.md).

GitHub main'e push ve başarılı CI, Railway'de çalışan sürümün kanıtı değildir.
Canlı `/api/version` üzerinde bu release ID ve yeni API'ler görülmeden yayın
tamamlandı sayılmamalı. Kalıcı volume yoksa wallet geçmişi redeploy'da silinebilir.
Cloud araştırma worker'ları ayrı, 30 Ekim 19:00 Londra bütçesiyle çalışır;
Railway deploy'u bu süreçleri otomatik oraya taşımaz.
