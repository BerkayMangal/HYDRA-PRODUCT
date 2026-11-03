# Telefonda HYDRA Spot denemesi

## Hazır olan

`feature/spot-wallet-preview-20261103` dalında `/spot/` ekranı:

- MetaMask/Rabby uygulama içi tarayıcılarında enjekte edilen cüzdan bağlantısı.
- Hyperliquid spot USDC bakiyesi ve UBTC/USDC işlem geçmişi.
- Güncel emir defteri, kişisel spot ücret oranı ve miktar üzerinden önizleme.
- Kullanıcının her emir için ayrı imzası; tek UBTC alım emri, IOC ve %0,5 kayma
  sınırı. 12–1.000 USDC tutar sınırı, komisyon payı ve 30 saniyelik önizleme süresi.
- Cüzdan/ağ değişiminde önizleme iptali; belirsiz gönderimde otomatik tekrar yok.

Özel anahtar, seed, API agent yetkisi, token allowance, para yatırma veya çekim
istenmiyor. Sunucuda emir imzalayan anahtar veya işlem endpoint'i eklenmedi.
UBTC, Bitcoin ağındaki yerel BTC değil, Unit üzerinden temsil edilen varlık.

## Doğrulama

- 238 Python testi geçti; spot route ve derlenmiş dosya erişimi dahil.
- 10 TypeScript testi geçti; token kimliği, hassas miktar/fiyat yuvarlama, komisyon
  rezervi, süresi dolan fiyat, yanlış imzalayan cüzdan ve kısmi sonuç ayrımı dahil.
- Emir özeti, resmi Hyperliquid Python SDK'sının `action_hash` fonksiyonuyla aynı
  sonucu verdi; nonce ve son geçerlilik zamanı da özete dahil.
- Chromium: 390px mobil görünüm, taşma yok, sahte sağlayıcıyla bağlantı/önizleme,
  yetersiz bakiye ve imza zinciri uyuşmazlığında gönderim engeli; tarayıcı hatası yok.
  Bu test gerçek cüzdana veya gerçek 100 USDC bakiyeye dair kanıt değildir.
- Canlı `/info` okumasında UBTC/USDC varlık kimliği 10142 doğrulandı. Tarayıcı CORS
  başlıkları ve `/exchange` OPTIONS ön kontrolü uygun. Gerçek exchange POST sayısı
  **sıfır**; gerçek telefon cüzdanında alım henüz doğrulanmadı.

Bazı cüzdanlar Hyperliquid'in `Exchange / Agent`, zincir kimliği 1337 olan imza
alanını reddeder. Böyle bir cüzdanda bağlantı ve önizleme kullanılabilir; gerçek
emir gönderilmez. Sırf bu engeli aşmak için sahte ağ veya geniş işlem yetkisi
isteyen API agent eklenmedi. WalletConnect proje kimliği yapılandırılmadı.

## Yayın durumu — başarılı

**3 Kasım 2026, yaklaşık 17:41 Londra/UTC:**
https://berkaymangal.github.io/HYDRA-PRODUCT/spot/

[Başarılı Pages yayını ve HTTPS/sürüm kontrolü](https://github.com/BerkayMangal/HYDRA-PRODUCT/actions/runs/41686433897).
[Main Python CI başarılı](https://github.com/BerkayMangal/HYDRA-PRODUCT/actions/runs/41686433903).
[Railway yayın işi beklendiği gibi atlandı](https://github.com/BerkayMangal/HYDRA-PRODUCT/actions/runs/41686591338).

İlk yayın site oluşturma yetkisinden, sonraki deneme Pages ortamının yalnızca
`main` dalına izin vermesinden dolayı reddedilmişti. Kullanıcı Pages kaynağını
GitHub Actions olarak etkinleştirdi. Ortam koruması korunarak yayın akışı `main`
dalına eklendi; gözden geçirilmiş UI kaynağı sabit commit'ten derleniyor.

- Yayın akışı commit'i: `3457ada` (`main`).
- UI kaynak commit'i: `ab44d58e756b7c1e6eb8fdcf68d65e49601400e7`.
- Yayın sonrası iş, `/spot/` HTML'ini ve `/spot/release.json` içindeki kaynak
  commit'ini HTTPS üzerinden doğruladı. Bu çalışma alanının ağ politikası Pages
  alan adına doğrudan okumaya izin vermiyor; doğrulama normal yayın işinde yapıldı.
- Build/test işi 10 TypeScript testi çalıştırdı. Main'deki ayrı Python CI geçti;
  main kod tabanında yerel olarak 186 test geçti. Feature dalındaki 238 testlik
  önceki sonuç ayrı kaynak revizyonuna aittir.
- Gerçek cüzdan imzası veya gerçek alım otomasyonda denenmedi.

Telefonda bu adresi MetaMask/Rabby uygulama içi tarayıcısında aç veya sayfadaki
MetaMask bağlantısını kullan. İlk adım bağlantı, spot bakiye ve alım önizlemesi.
Gerçek alım kullanıcının ayrı cüzdan imzasını gerektirir; imza alanı uyumluluğu
sınırlamaları yukarıda açıklanmıştır.

Railway üretim uygulaması bu yayınla güncellenmedi. Çalışan BTC/ETH model servisi
17:40 UTC kontrolünde HTTP 200, `state=active`, `error=null` döndürdü. Bu checkout'ta
bulunmayan model kaynakları ve `/assets/hydra.js` nedeniyle uygulamayı eski kodla
değiştirecek bir dağıtım yapılmadı. Yalnız spot yayın commit'leri `[spot-preview]`
önekini kullanır; Railway otomatik yayın koşulu bunları dışlar, Python CI çalışır.

## Gelir modeli sorusu

Hyperliquid **Builder Code / Builder Fee**, kullanıcı tarafından onaylanan
azami ücret ve emre eklenen builder adresi/ücret alanıyla çalışıyor. **Referral
Code** ayrı bir yönlendirme programı. Sadece cüzdan bağlanması ücret doğurmuyor.
Resmi SDK örneği:
https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/examples/basic_builder_fee.py

Bu sürüm hiçbir builder ücreti veya builder ücret onayı istemiyor. Spot alım
özelindeki güncel gelir uygunluğu ve tahsil kuralları bağımsız resmi dokümanla
henüz doğrulanmadığı için alım başına gelir sözü veya gelir tahmini yapılmadı.
Bu soru sebebiyle kullanıcı hesabına referral/builder onayı gönderilmedi.
