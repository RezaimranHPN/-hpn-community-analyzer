# HPN Community Analyzer (MVP)

Aplikasi lokal untuk menganalisis file `.txt` hasil Export Chat WhatsApp dan menghasilkan ranking member serta laporan Excel.

## Menjalankan aplikasi

Perlu Python 3.10 atau lebih baru.

1. Ekstrak ZIP ini.
2. Buka Terminal / Command Prompt pada folder ini.
3. Instal dependensi:

   `pip install -r requirements.txt`

4. Jalankan:

   `streamlit run app.py`

5. Browser akan membuka aplikasi. Upload file `.txt` ekspor WhatsApp.

## Fitur MVP

- Membaca beberapa format umum ekspor WhatsApp Android/iOS.
- Menghitung pesan, hari aktif, pesan berulang, dan sinyal kontribusi.
- Ranking aktivitas, konsistensi, dan skor kontribusi heuristik.
- Melihat pesan yang terindikasi spam/duplikat.
- Mengunduh laporan Excel.

## Catatan penting

- File diproses oleh aplikasi lokal; versi ini tidak mengirim percakapan ke layanan AI eksternal.
- Deteksi spam dan kontribusi bersifat heuristik, bukan penilaian pasti.
- Format ekspor WhatsApp bisa berbeda berdasarkan perangkat/bahasa. Uji dengan file nyata dan periksa apakah tanggal serta nama pengirim terbaca benar.
- Untuk analisis kualitas berbasis AI yang lebih mendalam, integrasi model AI dan pengamanan data perlu ditambahkan secara terpisah.
- Jangan gunakan ranking otomatis sebagai satu-satunya dasar untuk memberi reward atau menilai member.
