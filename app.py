import re
from datetime import datetime, date
from io import BytesIO

import pandas as pd
import streamlit as st


# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================

st.set_page_config(
    page_title="HPN Community Analyzer",
    page_icon="💬",
    layout="wide"
)

st.title("💬 HPN Community Analyzer")
st.caption(
    "Community Intelligence Dashboard | "
    "Analisis aktivitas, konsistensi, dan kontribusi anggota HPN"
)

st.info(
    "Privasi: aplikasi ini menggunakan kode analisis berbasis aturan, "
    "tanpa mengirim percakapan ke layanan AI eksternal. "
    "Hindari mengunggah percakapan sensitif. "
    "Jika menggunakan hosting online, pemrosesan dilakukan di lingkungan hosting."
)


# =========================================================
# 2. POLA FORMAT EKSPOR WHATSAPP
# =========================================================

TIME_PATTERN = (
    r"\d{1,2}[.:]\d{2}"
    r"(?:[.:]\d{2})?"
    r"(?:\s?[APap][Mm])?"
)

HEADER_PATTERNS = [
    re.compile(
        rf"^\[(\d{{1,2}}/\d{{1,2}}/\d{{2,4}}),\s*"
        rf"({TIME_PATTERN})\]\s(.*)$"
    ),
    re.compile(
        rf"^(\d{{1,2}}/\d{{1,2}}/\d{{2,4}}),\s*"
        rf"({TIME_PATTERN})\s-\s(.*)$"
    ),
    re.compile(
        rf"^(\d{{4}}-\d{{1,2}}-\d{{1,2}}),\s*"
        rf"({TIME_PATTERN})\s-\s(.*)$"
    ),
]

SYSTEM_PHRASES = [
    "messages and calls are end-to-end encrypted",
    "created the group",
    "created group",
    "added ",
    "removed ",
    "changed the subject",
    "changed this group's icon",
    "changed the group description",
    "left",
    "joined using this group's invite link",
    "you're now an admin",
    "this message was deleted",
    "security code changed",
    "changed their phone number",
]

SPAM_PHRASES = [
    "forwarded many times",
    "klik link",
    "click here",
    "daftar sekarang",
    "promo terbatas",
    "beli sekarang",
    "hubungi wa",
    "bit.ly/",
    "tinyurl.com",
    "wa.me/",
]

HELPFUL_TERMS = [
    "terima kasih",
    "makasih",
    "semoga",
    "menurutku",
    "pengalaman",
    "aku pernah",
    "saya pernah",
    "caranya",
    "coba",
    "bisa dibantu",
    "jawabannya",
    "informasi",
    "bermanfaat",
    "setuju",
    "izin berbagi",
    "sharing",
    "rekomendasi",
    "tips",
]

QUESTION_MARKERS = [
    "?",
    "bagaimana",
    "gimana",
    "kenapa",
    "apakah",
    "ada yang tahu",
    "minta saran",
]

LOW_SIGNAL = {
    "ok", "oke", "iya", "ya", "wkwk", "haha", "hehe",
    "amin", "👍", "🙏", "😂", "❤️", "siap", "noted"
}


# =========================================================
# 3. PARSER TANGGAL DAN WAKTU
# =========================================================

def parse_datetime(date_s, time_s):
    t = time_s.strip().replace(".", ":")

    t = (
        t.replace(" a.m.", " AM")
         .replace(" p.m.", " PM")
         .replace(" a.m", " AM")
         .replace(" p.m", " PM")
    )

    formats = [
        "%d/%m/%y %H:%M:%S",
        "%d/%m/%y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%y %I:%M:%S %p",
        "%m/%d/%y %I:%M %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%y %H:%M:%S",
        "%m/%d/%y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(f"{date_s} {t}", fmt)
        except ValueError:
            continue

    return None


def split_sender_message(content):
    if ": " in content:
        sender, message = content.split(": ", 1)
        return sender.strip(), message.strip()

    # Pesan sistem yang tidak mempunyai pemisah nama dan pesan.
    return "System", content.strip()


# =========================================================
# 4. MEMBACA FILE WHATSAPP
# =========================================================

def parse_chat(raw_text):
    rows = []
    current = None

    for line in raw_text.splitlines():
        match = None

        for pattern in HEADER_PATTERNS:
            match = pattern.match(line)

            if match:
                break

        if match:
            if current is not None:
                rows.append(current)

            date_s, time_s, content = match.groups()
            sender, message = split_sender_message(content)

            current = {
                "datetime": parse_datetime(date_s, time_s),
                "date_raw": date_s,
                "time_raw": time_s,
                "sender": sender,
                "message": message,
            }

        elif current is not None:
            # Mempertahankan pesan yang berlanjut ke baris berikutnya.
            current["message"] += "\n" + line

    if current is not None:
        rows.append(current)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["message"] = df["message"].fillna("").astype(str)
    df["sender"] = df["sender"].fillna("").astype(str)

    def detect_system(row):
        sender = row["sender"].strip().lower()
        message = row["message"].strip().lower()

        if sender in ("", "system"):
            return True

        return (
            any(phrase in message for phrase in SYSTEM_PHRASES)
            and len(message) < 300
        )

    df["is_system"] = df.apply(detect_system, axis=1)

    normalized = (
        df["message"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["normalized"] = normalized

    df["is_duplicate"] = (
        normalized.ne("")
        & normalized.duplicated(keep=False)
    )

    df["is_spam_suspect"] = (
        df["message"].str.lower().apply(
            lambda m: any(p in m for p in SPAM_PHRASES)
        )
        | df["is_duplicate"]
    )

    df["word_count"] = (
        df["message"].str.findall(r"\b[\w'-]+\b").str.len()
    )

    df["is_low_signal"] = normalized.apply(
        lambda m: m in LOW_SIGNAL
        or (len(m) <= 3 and "?" not in m)
    )

    df["has_question"] = df["message"].str.lower().apply(
        lambda m: any(q in m for q in QUESTION_MARKERS)
    )

    df["helpful_signal"] = df["message"].str.lower().apply(
        lambda m: any(t in m for t in HELPFUL_TERMS)
    )

    df["is_substantive"] = (
        (df["word_count"] >= 8)
        | df["has_question"]
        | df["helpful_signal"]
    )

    return df


# =========================================================
# 5. MEMBUAT RANKING MEMBER
# =========================================================

def make_summary(df):
    messages = df[
        (~df["is_system"])
        & df["sender"].ne("")
        & df["datetime"].notna()
    ].copy()

    if messages.empty:
        return pd.DataFrame()

    summary = messages.groupby("sender").agg(
        total_messages=("message", "size"),
        active_days=(
            "datetime",
            lambda x: x.dt.date.nunique()
        ),
        first_message=("datetime", "min"),
        last_message=("datetime", "max"),
        duplicate_messages=("is_duplicate", "sum"),
        spam_suspects=("is_spam_suspect", "sum"),
        low_signal_messages=("is_low_signal", "sum"),
        substantive_messages=("is_substantive", "sum"),
        questions=("has_question", "sum"),
        helpful_signals=("helpful_signal", "sum"),
        total_words=("word_count", "sum"),
    ).reset_index().rename(
        columns={"sender": "member"}
    )

    summary["nonspam_estimate"] = (
        summary["total_messages"] - summary["spam_suspects"]
    ).clip(lower=0)

    summary["spam_rate_pct"] = (
        summary["spam_suspects"]
        / summary["total_messages"].clip(lower=1)
        * 100
    ).round(1)

    max_days = max(int(summary["active_days"].max()), 1)
    max_messages = max(int(summary["nonspam_estimate"].max()), 1)
    max_substantive = max(
        int(summary["substantive_messages"].max()), 1
    )
    max_helpful = max(
        int(summary["helpful_signals"].max()), 1
    )

    consistency = summary["active_days"] / max_days * 100
    activity = summary["nonspam_estimate"] / max_messages * 100
    substance = (
        summary["substantive_messages"] / max_substantive * 100
    )
    helpful = summary["helpful_signals"] / max_helpful * 100

    spam_penalty = (
        summary["spam_rate_pct"] * 0.5
    ).clip(upper=35)

    summary["contribution_score"] = (
        0.25 * activity
        + 0.25 * consistency
        + 0.30 * substance
        + 0.20 * helpful
        - spam_penalty
    ).clip(0, 100).round(1)

    summary["activity_rank"] = (
        summary["nonspam_estimate"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    summary["consistency_rank"] = (
        summary["active_days"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    summary["contribution_rank"] = (
        summary["contribution_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return summary.sort_values(
        ["contribution_rank", "nonspam_estimate"],
        ascending=[True, False]
    )


# =========================================================
# 6. RINGKASAN BULANAN
# =========================================================

def make_monthly_summary(df):
    messages = df[
        (~df["is_system"])
        & df["sender"].ne("")
        & df["datetime"].notna()
    ].copy()

    if messages.empty:
        return pd.DataFrame()

    messages["bulan"] = (
        messages["datetime"].dt.to_period("M").astype(str)
    )

    monthly = messages.groupby("bulan").agg(
        total_pesan=("message", "size"),
        jumlah_member=("sender", "nunique"),
        hari_aktif_grup=(
            "datetime",
            lambda x: x.dt.date.nunique()
        ),
        pesan_terindikasi_spam=("is_spam_suspect", "sum"),
        pesan_substantif=("is_substantive", "sum"),
    ).reset_index()

    monthly["pesan_nonspam_perkiraan"] = (
        monthly["total_pesan"] - monthly["pesan_terindikasi_spam"]
    ).clip(lower=0)

    monthly["rata_pesan_per_member"] = (
        monthly["total_pesan"]
        / monthly["jumlah_member"].clip(lower=1)
    ).round(1)

    return monthly.sort_values("bulan")


# =========================================================
# 7. UPLOAD FILE
# =========================================================

uploaded = st.file_uploader(
    "Upload file ekspor WhatsApp (.txt)",
    type=["txt"]
)

if uploaded:

    try:
        raw = uploaded.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        raw = uploaded.getvalue().decode(
            "utf-8", errors="replace"
        )

    df_all = parse_chat(raw)

    if df_all.empty:
        st.error(
            "Format file belum berhasil dibaca. "
            "Pastikan file berasal dari Export Chat WhatsApp."
        )
        st.code(
            "[24/08/24, 07.11.35] Nama Member: Isi pesan"
        )
        st.stop()

    valid_dates = df_all["datetime"].dropna()

    st.caption(
        f"Baris percakapan terdeteksi: {len(df_all):,} | "
        f"Tanggal/waktu terbaca: {len(valid_dates):,}"
    )

    if valid_dates.empty:
        st.error(
            "Baris ditemukan, tetapi tanggal dan waktu "
            "belum berhasil diproses."
        )
        st.stop()

    # =====================================================
    # 8. FILTER PERIODE
    # =====================================================

    st.divider()
    st.subheader("📅 Pilih Periode Analisis")

    period_mode = st.radio(
        "Periode yang ingin dianalisis",
        [
            "Semua periode",
            "Bulan tertentu",
            "Rentang tanggal",
        ],
        horizontal=True
    )

    df = df_all.copy()

    if period_mode == "Bulan tertentu":

        months = sorted(
            valid_dates.dt.to_period("M").unique(),
            reverse=True
        )

        month_labels = [
            m.strftime("%B %Y") for m in months
        ]

        selected_label = st.selectbox(
            "Pilih bulan",
            month_labels
        )

        selected_index = month_labels.index(selected_label)
        selected_month = months[selected_index]

        df = df[
            df["datetime"].notna()
            & (
                df["datetime"].dt.to_period("M")
                == selected_month
            )
        ].copy()

    elif period_mode == "Rentang tanggal":

        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        col_start, col_end = st.columns(2)

        with col_start:
            start_date = st.date_input(
                "Tanggal awal",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )

        with col_end:
            end_date = st.date_input(
                "Tanggal akhir",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )

        if start_date > end_date:
            st.error(
                "Tanggal awal tidak boleh melewati tanggal akhir."
            )
            st.stop()

        df = df[
            df["datetime"].notna()
            & (
                df["datetime"].dt.date >= start_date
            )
            & (
                df["datetime"].dt.date <= end_date
            )
        ].copy()

    if df.empty:
        st.warning(
            "Tidak ditemukan pesan pada periode yang dipilih."
        )
        st.stop()

    # =====================================================
    # 9. RANKING SESUAI PERIODE PILIHAN
    # =====================================================

    summary = make_summary(df)

    if summary.empty:
        st.warning(
            "Tidak ada pesan anggota yang dapat dianalisis "
            "pada periode ini."
        )
        st.stop()

    start = df["datetime"].min()
    end = df["datetime"].max()

    st.success(
        f"Periode yang dianalisis: "
        f"{start:%d %b %Y} – {end:%d %b %Y}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Member terdeteksi",
        f"{summary['member'].nunique():,}"
    )

    c2.metric(
        "Total pesan anggota",
        f"{int(summary['total_messages'].sum()):,}"
    )

    c3.metric(
        "Hari aktif grup",
        f"{df.loc[~df['is_system'], 'datetime'].dt.date.nunique():,}"
    )

    c4.metric(
        "Indikasi spam/duplikat",
        f"{int(summary['spam_suspects'].sum()):,}"
    )

    st.caption(
        "Pesan yang sama bisa saja wajar. "
        "Indikasi spam perlu diperiksa sebelum mengambil keputusan."
    )

    st.divider()
    st.subheader("🏆 Ranking Member")

    ranking_type = st.selectbox(
        "Urutkan ranking berdasarkan",
        [
            "Kontribusi keseluruhan",
            "Jumlah pesan nonspam (perkiraan)",
            "Hari aktif",
            "Jumlah pesan",
            "Pesan substantif",
        ]
    )

    if ranking_type == "Jumlah pesan nonspam (perkiraan)":
        shown = summary.sort_values(
            ["nonspam_estimate", "active_days"],
            ascending=False
        ).copy()

    elif ranking_type == "Hari aktif":
        shown = summary.sort_values(
            ["active_days", "nonspam_estimate"],
            ascending=False
        ).copy()

    elif ranking_type == "Jumlah pesan":
        shown = summary.sort_values(
            ["total_messages", "active_days"],
            ascending=False
        ).copy()

    elif ranking_type == "Pesan substantif":
        shown = summary.sort_values(
            ["substantive_messages", "helpful_signals"],
            ascending=False
        ).copy()

    else:
        shown = summary.sort_values(
            ["contribution_score", "nonspam_estimate"],
            ascending=False
        ).copy()

    top_n = st.selectbox(
        "Jumlah member yang ditampilkan",
        ["Semua member", "Top 10", "Top 20", "Top 50"]
    )

    if top_n != "Semua member":
        shown = shown.head(int(top_n.split()[1]))

    display = shown[
        [
            "member",
            "total_messages",
            "nonspam_estimate",
            "active_days",
            "spam_suspects",
            "spam_rate_pct",
            "substantive_messages",
            "helpful_signals",
            "contribution_score",
        ]
    ].rename(columns={
        "member": "Member",
        "total_messages": "Total Pesan",
        "nonspam_estimate": "Pesan Nonspam (perkiraan)",
        "active_days": "Hari Aktif",
        "spam_suspects": "Indikasi Spam/Duplikat",
        "spam_rate_pct": "Indikasi Spam (%)",
        "substantive_messages": "Pesan Substantif",
        "helpful_signals": "Sinyal Membantu",
        "contribution_score": "Skor Kontribusi",
    })

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📊 Aktivitas Member")
    st.bar_chart(
        shown.set_index("member")["nonspam_estimate"].head(20)
    )

    # =====================================================
    # 10. PERBANDINGAN AKTIVITAS ANTARBULAN
    # =====================================================

    st.divider()
    st.subheader("📈 Perbandingan Aktivitas Antarbulan")

    monthly = make_monthly_summary(df_all)

    if not monthly.empty:
        st.caption(
            "Bagian ini memakai seluruh data yang diunggah, "
            "bukan hanya periode ranking yang dipilih."
        )

        st.dataframe(
            monthly.rename(columns={
                "bulan": "Bulan",
                "total_pesan": "Total Pesan",
                "jumlah_member": "Member Aktif",
                "hari_aktif_grup": "Hari Aktif Grup",
                "pesan_terindikasi_spam": "Indikasi Spam/Duplikat",
                "pesan_substantif": "Pesan Substantif",
                "pesan_nonspam_perkiraan": "Pesan Nonspam (perkiraan)",
                "rata_pesan_per_member": "Rata-rata Pesan per Member",
            }),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            monthly.set_index("bulan")[
                ["total_pesan", "pesan_nonspam_perkiraan"]
            ]
        )

    # =====================================================
    # 11. RINCIAN PESAN
    # =====================================================

    st.divider()
    st.subheader("🔎 Pemeriksaan Pesan")

    with st.expander(
        "Lihat pesan yang terindikasi spam atau duplikat"
    ):
        flagged = df[
            (~df["is_system"])
            & df["is_spam_suspect"]
        ][
            [
                "datetime",
                "sender",
                "message",
                "is_duplicate"
            ]
        ]

        st.dataframe(
            flagged,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("Lihat seluruh pesan dalam periode pilihan"):
        st.dataframe(
            df[
                [
                    "datetime",
                    "sender",
                    "message",
                    "is_system"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # 12. EKSPOR EXCEL
    # =====================================================

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        summary.to_excel(
            writer,
            sheet_name="Ranking Periode",
            index=False
        )

        df.to_excel(
            writer,
            sheet_name="Pesan Periode",
            index=False
        )

        monthly.to_excel(
            writer,
            sheet_name="Perbandingan Bulanan",
            index=False
        )

        flagged = df[
            (~df["is_system"])
            & df["is_spam_suspect"]
        ][
            [
                "datetime",
                "sender",
                "message",
                "is_duplicate"
            ]
        ]

        flagged.to_excel(
            writer,
            sheet_name="Periksa Spam",
            index=False
        )

    st.download_button(
        "⬇️ Download Laporan Excel",
        data=output.getvalue(),
        file_name="HPN_Community_Analysis.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

else:

    st.markdown("### Cara Menggunakan")

    st.markdown(
        "1. Buka grup WhatsApp HPN.\n"
        "2. Pilih **Ekspor chat → Tanpa media**.\n"
        "3. Upload file `.txt`.\n"
        "4. Pilih periode analisis.\n"
        "5. Periksa ranking dan perbandingan aktivitas.\n"
        "6. Download laporan Excel."
    )

    st.markdown("### Indikator Ranking")

    st.markdown(
        "- **Aktivitas:** jumlah pesan setelah dikurangi "
        "pesan yang terindikasi spam/duplikat.\n"
        "- **Konsistensi:** jumlah hari aktif.\n"
        "- **Kontribusi:** estimasi berbasis aktivitas, "
        "konsistensi, pesan substantif, dan sinyal membantu."
    )

    st.warning(
        "Versi ini menggunakan heuristik, bukan model AI yang "
        "memahami konteks percakapan secara mendalam. "
        "Hasil ranking sebaiknya menjadi bahan evaluasi, "
        "bukan satu-satunya dasar menentukan kontribusi member."
    )
