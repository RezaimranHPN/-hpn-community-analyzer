import re
import io
import unicodedata
from datetime import datetime, date

import pandas as pd
import streamlit as st


# ============================================================
# 1. KONFIGURASI APLIKASI
# ============================================================

st.set_page_config(
    page_title="HPN Community Analyzer",
    page_icon="💬",
    layout="wide"
)

st.title("💬 HPN Community Analyzer")
st.caption(
    "Community Intelligence Dashboard | "
    "Activity, Contribution, and Moderate Spam Detection"
)


# ============================================================
# 2. ATURAN DETEKSI PESAN
# ============================================================

SYSTEM_PHRASES = [
    "messages and calls are end-to-end encrypted",
    "created the group",
    "added you",
    "added ",
    "removed ",
    "left",
    "changed the group description",
    "changed this group's icon",
    "changed the group icon",
    "changed the subject",
    "joined using this group's invite link",
    "joined using this group invite link",
    "security code changed",
    "disappearing messages",
    "turned on disappearing messages",
    "turned off disappearing messages",
    "changed the group settings",
    "this message was deleted",
    "you deleted this message",
    "pinned a message",
    "changed the group name",
    "created this group"
]

PROMO_PHRASES = [
    "promo terbatas",
    "diskon besar",
    "beli sekarang",
    "daftar sekarang",
    "buruan daftar",
    "jangan sampai ketinggalan",
    "harga spesial",
    "free ongkir",
    "gratis ongkir",
    "kode referral",
    "kode promo",
    "affiliate",
    "reseller",
    "dropship",
    "order sekarang",
    "cek keranjang",
    "link pembelian",
    "hubungi wa",
    "hubungi saya untuk order",
    "dm untuk order",
    "open reseller",
    "open jastip"
]

HELPFUL_PHRASES = [
    "semoga membantu",
    "terima kasih informasinya",
    "terima kasih infonya",
    "aku bantu",
    "saya bantu",
    "coba cek",
    "bisa dicoba",
    "menurut pengalaman",
    "pengalaman saya",
    "berdasarkan pengalaman",
    "aku sarankan",
    "saya sarankan",
    "semoga bermanfaat",
    "ini caranya",
    "berikut caranya",
    "izin berbagi",
    "izin share",
    "aku pernah mengalami",
    "saya pernah mengalami"
]

URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|"
    r"\bbit\.ly/\S+|\btinyurl\.com/\S+|\bwa\.me/\S+)",
    re.IGNORECASE
)

WHATSAPP_HEADER_PATTERN = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*"
    r"(\d{1,2}[.:]\d{2}(?:[.:]\d{2})?)\]\s*(.*)$"
)


# ============================================================
# 3. FUNGSI NORMALISASI
# ============================================================

def normalize_text(text):
    """
    Menyamakan format teks agar pesan yang sama
    tetap terdeteksi meskipun berbeda kapitalisasi
    atau tanda baca.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def contains_phrase(text, phrases):
    text_lower = str(text).lower()

    return any(
        phrase in text_lower
        for phrase in phrases
    )


# ============================================================
# 4. PARSER EKSPOR WHATSAPP
# ============================================================

def parse_whatsapp_chat(uploaded_file):
    """
    Membaca ekspor chat WhatsApp .txt.

    Mendukung format waktu:
    07.11
    07.11.35
    07:11
    07:11:35

    Pesan multiline akan digabungkan ke pesan sebelumnya.
    """

    raw_bytes = uploaded_file.getvalue()

    try:
        raw_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("utf-8", errors="replace")

    lines = raw_text.splitlines()

    messages = []
    current_message = None

    def save_current_message():
        if current_message is not None:
            messages.append(current_message.copy())

    for line in lines:
        match = WHATSAPP_HEADER_PATTERN.match(line)

        if match:
            save_current_message()

            date_text = match.group(1)
            time_text = match.group(2)
            body = match.group(3)

            sender = "Unknown"
            message = body

            # Pisahkan nama pengirim dari isi pesan.
            # Contoh:
            # GCK Siti Rita: Selamat pagi
            # HPN | General: Pesan grup

            if ": " in body:
                possible_sender, possible_message = body.split(
                    ": ", 1
                )

                if possible_sender.strip():
                    sender = possible_sender.strip()
                    message = possible_message.strip()

            # Konversi format tanggal.
            parsed_datetime = None

            date_formats = [
                "%d/%m/%y",
                "%d/%m/%Y"
            ]

            time_normalized = time_text.replace(".", ":")

            time_formats = [
                "%H:%M:%S",
                "%H:%M"
            ]

            for date_format in date_formats:
                for time_format in time_formats:
                    try:
                        parsed_datetime = datetime.strptime(
                            f"{date_text} {time_normalized}",
                            f"{date_format} {time_format}"
                        )
                        break
                    except ValueError:
                        continue

                if parsed_datetime is not None:
                    break

            current_message = {
                "datetime": parsed_datetime,
                "sender": sender,
                "message": message
            }

        else:
            # Baris lanjutan dari pesan multiline.
            if current_message is not None:
                current_message["message"] += "\n" + line

    save_current_message()

    df = pd.DataFrame(messages)

    if df.empty:
        return df

    df["message"] = df["message"].fillna("").astype(str)
    df["sender"] = df["sender"].fillna("Unknown").astype(str)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df = df.dropna(subset=["datetime"]).copy()

    df = df.sort_values("datetime").reset_index(drop=True)

    return df


# ============================================================
# 5. PEMBERSIHAN DAN KLASIFIKASI PESAN
# ============================================================

def classify_messages(df):
    """
    Menambahkan indikator:
    - pesan sistem
    - duplikat
    - promosi mencurigakan
    - pesan substantif
    - indikasi membantu
    """

    df = df.copy()

    df["normalized_message"] = df["message"].apply(
        normalize_text
    )

    # Pesan sistem tidak dihitung sebagai kontribusi anggota.
    df["is_system"] = df["message"].apply(
        lambda text: contains_phrase(text, SYSTEM_PHRASES)
    )

    # Pesan kosong tidak dihitung.
    df["is_empty"] = df["normalized_message"].eq("")

    # Cari pesan identik dari anggota yang sama.
    # Kemunculan pertama juga ditandai jika pesan diulang.
    df["duplicate_count"] = 0

    eligible = (
        ~df["is_system"]
        & ~df["is_empty"]
    )

    duplicate_counts = (
        df.loc[eligible]
        .groupby(["sender", "normalized_message"])["message"]
        .transform("size")
    )

    df.loc[eligible, "duplicate_count"] = duplicate_counts

    df["is_duplicate"] = (
        eligible
        & (df["duplicate_count"] > 1)
    )

    # Indikasi promosi.
    df["has_promo_phrase"] = df["message"].apply(
        lambda text: contains_phrase(text, PROMO_PHRASES)
    )

    df["has_url"] = df["message"].apply(
        lambda text: bool(URL_PATTERN.search(str(text)))
    )

    # Pesan dengan tautan atau kata promosi perlu ditinjau.
    df["is_promo_suspect"] = (
        ~df["is_system"]
        & ~df["is_empty"]
        & (
            df["has_promo_phrase"]
            | df["has_url"]
        )
    )

    # Pesan substantif: indikator sederhana, bukan penilaian AI.
    df["word_count"] = df["message"].apply(
        lambda text: len(str(text).split())
    )

    df["is_substantive"] = (
        ~df["is_system"]
        & ~df["is_empty"]
        & (
            (df["word_count"] >= 8)
            | df["message"].str.contains(
                r"\?",
                regex=True,
                na=False
            )
        )
    )

    df["has_helpful_signal"] = df["message"].apply(
        lambda text: contains_phrase(text, HELPFUL_PHRASES)
    )

    df.loc[
        df["is_system"] | df["is_empty"],
        "is_substantive"
    ] = False

    df.loc[
        df["is_system"] | df["is_empty"],
        "has_helpful_signal"
    ] = False

    # Klasifikasi awal untuk pemeriksaan.
    df["review_reason"] = ""

    df.loc[
        df["is_duplicate"],
        "review_reason"
    ] = "Pesan identik berulang"

    df.loc[
        df["is_promo_suspect"],
        "review_reason"
    ] = df.loc[
        df["is_promo_suspect"],
        "review_reason"
    ].apply(
        lambda current: (
            current + "; " if current else ""
        ) + "Promosi atau tautan mencurigakan"
    )

    df["needs_review"] = (
        df["is_duplicate"]
        | df["is_promo_suspect"]
    )

    return df


# ============================================================
# 6. MODERATE SPAM DETECTION
# ============================================================

def apply_moderate_spam_rules(df):
    """
    Aturan moderat:
    1. Duplikat saja tidak langsung terkena penalti.
    2. Satu tautan saja tidak langsung terkena penalti.
    3. Pola duplikat yang kuat atau promosi berulang
       dapat masuk kategori indikasi spam kuat.

    Aturan diterapkan pada periode data yang sedang dianalisis.
    """

    df = df.copy()

    df["strong_spam_pattern"] = False
    df["strong_spam_reason"] = ""

    eligible = (
        ~df["is_system"]
        & ~df["is_empty"]
    )

    # Hitung pesan identik per anggota.
    duplicate_group_counts = (
        df.loc[eligible]
        .groupby(["sender", "normalized_message"])["message"]
        .transform("size")
    )

    df["same_message_count"] = 0

    df.loc[
        eligible,
        "same_message_count"
    ] = duplicate_group_counts

    # Aturan 1:
    # Pesan identik yang dikirim 3 kali atau lebih
    # dalam periode analisis.
    strong_duplicate = (
        eligible
        & (df["same_message_count"] >= 3)
    )

    df.loc[
        strong_duplicate,
        "strong_spam_pattern"
    ] = True

    df.loc[
        strong_duplicate,
        "strong_spam_reason"
    ] = "Pesan identik dikirim 3 kali atau lebih"

    # Aturan 2:
    # Promosi/tautan mencurigakan berulang minimal 3 kali
    # oleh anggota yang sama dalam periode analisis.
    promo_counts = (
        df.loc[
            eligible & df["is_promo_suspect"]
        ]
        .groupby("sender")["message"]
        .transform("size")
    )

    df["promo_suspect_count_by_sender"] = 0

    df.loc[
        eligible & df["is_promo_suspect"],
        "promo_suspect_count_by_sender"
    ] = promo_counts

    strong_promo = (
        eligible
        & df["is_promo_suspect"]
        & (df["promo_suspect_count_by_sender"] >= 3)
    )

    df.loc[
        strong_promo,
        "strong_spam_pattern"
    ] = True

    df.loc[
        strong_promo,
        "strong_spam_reason"
    ] = df.loc[
        strong_promo,
        "strong_spam_reason"
    ].apply(
        lambda current: (
            current + "; " if current else ""
        ) + "Promosi atau tautan mencurigakan berulang"
    )

    # Hanya pesan dengan pola kuat yang masuk hitungan penalti.
    df["spam_penalty_flag"] = (
        df["strong_spam_pattern"]
        & ~df["is_system"]
        & ~df["is_empty"]
    )

    # Status pemeriksaan yang lebih jelas.
    df["spam_status"] = "Normal"

    df.loc[
        df["needs_review"],
        "spam_status"
    ] = "Perlu ditinjau"

    df.loc[
        df["spam_penalty_flag"],
        "spam_status"
    ] = "Indikasi spam kuat"

    return df


# ============================================================
# 7. MEMBANGUN RANKING ANGGOTA
# ============================================================

def build_member_ranking(df):
    """
    Menghasilkan:
    - Activity Score
    - Contribution Score
    - indikator aktivitas dan kontribusi

    Skor adalah indikator relatif berdasarkan data yang tersedia.
    """

    valid = df[
        ~df["is_system"]
        & ~df["is_empty"]
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    ranking = (
        valid.groupby("sender")
        .agg(
            total_messages=("message", "count"),
            active_days=(
                "datetime",
                lambda values: values.dt.date.nunique()
            ),
            substantive_messages=("is_substantive", "sum"),
            helpful_signals=("has_helpful_signal", "sum"),
            duplicate_messages=("is_duplicate", "sum"),
            promo_suspects=("is_promo_suspect", "sum"),
            spam_penalty_messages=("spam_penalty_flag", "sum")
        )
        .reset_index()
        .rename(columns={"sender": "member"})
    )

    total_days = max(
        1,
        valid["datetime"].dt.date.nunique()
    )

    # Persentase pesan substantif terhadap semua pesan anggota.
    ranking["substantive_rate"] = (
        ranking["substantive_messages"]
        / ranking["total_messages"].clip(lower=1)
    )

    ranking["helpful_rate"] = (
        ranking["helpful_signals"]
        / ranking["total_messages"].clip(lower=1)
    )

    ranking["spam_penalty_rate"] = (
        ranking["spam_penalty_messages"]
        / ranking["total_messages"].clip(lower=1)
    )

    # Normalisasi jumlah pesan dan hari aktif ke skala 0-100.
    max_messages = max(
        1,
        ranking["total_messages"].max()
    )

    max_active_days = max(
        1,
        ranking["active_days"].max()
    )

    ranking["activity_volume_score"] = (
        ranking["total_messages"] / max_messages
    ) * 100

    ranking["consistency_score"] = (
        ranking["active_days"] / total_days
    ) * 100

    # Activity Score:
    # 60% volume pesan, 40% konsistensi.
    ranking["activity_score"] = (
        ranking["activity_volume_score"] * 0.60
        + ranking["consistency_score"] * 0.40
    )

    # Skor kontribusi berbasis indikator:
    # 50% pesan substantif
    # 30% sinyal membantu
    # 20% konsistensi aktivitas
    # Penalti diterapkan hanya untuk pola spam kuat.
    ranking["contribution_base_score"] = (
        ranking["substantive_rate"] * 50
        + ranking["helpful_rate"] * 30
        + (ranking["active_days"] / total_days) * 20
    )

    # Penalti maksimal 30 poin.
    # Duplikat biasa atau satu tautan tidak langsung dihukum.
    ranking["spam_penalty_points"] = (
        ranking["spam_penalty_rate"] * 30
    ).clip(upper=30)

    ranking["contribution_score"] = (
        ranking["contribution_base_score"]
        - ranking["spam_penalty_points"]
    ).clip(lower=0, upper=100)

    ranking["activity_score"] = (
        ranking["activity_score"]
        .clip(lower=0, upper=100)
        .round(2)
    )

    ranking["contribution_score"] = (
        ranking["contribution_score"].round(2)
    )

    ranking["spam_penalty_points"] = (
        ranking["spam_penalty_points"].round(2)
    )

    ranking["substantive_rate_pct"] = (
        ranking["substantive_rate"] * 100
    ).round(1)

    ranking["helpful_rate_pct"] = (
        ranking["helpful_rate"] * 100
    ).round(1)

    ranking["spam_penalty_rate_pct"] = (
        ranking["spam_penalty_rate"] * 100
    ).round(1)

    # Ranking terpisah.
    ranking["activity_rank"] = (
        ranking["activity_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    ranking["contribution_rank"] = (
        ranking["contribution_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    ranking = ranking.sort_values(
        ["contribution_score", "activity_score"],
        ascending=False
    ).reset_index(drop=True)

    return ranking


# ============================================================
# 8. RINGKASAN BULANAN
# ============================================================

def build_monthly_summary(df):
    valid = df[
        ~df["is_system"]
        & ~df["is_empty"]
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    valid["bulan"] = (
        valid["datetime"]
        .dt.to_period("M")
        .astype(str)
    )

    summary = (
        valid.groupby("bulan")
        .agg(
            total_pesan=("message", "count"),
            jumlah_member=("sender", "nunique"),
            hari_aktif_grup=(
                "datetime",
                lambda values: values.dt.date.nunique()
            ),
            pesan_perlu_ditinjau=("needs_review", "sum"),
            indikasi_spam_kuat=("spam_penalty_flag", "sum"),
            pesan_substantif=("is_substantive", "sum")
        )
        .reset_index()
    )

    summary["rata_pesan_per_member"] = (
        summary["total_pesan"]
        / summary["jumlah_member"].clip(lower=1)
    ).round(2)

    return summary


# ============================================================
# 9. FILTER PERIODE
# ============================================================

def filter_period(df):
    min_date = df["datetime"].min().date()
    max_date = df["datetime"].max().date()

    st.subheader("🗓️ Filter Periode Analisis")

    mode = st.radio(
        "Pilih periode",
        [
            "Semua Data",
            "Pilih Bulan",
            "Rentang Tanggal"
        ],
        horizontal=True
    )

    if mode == "Semua Data":
        filtered = df.copy()

    elif mode == "Pilih Bulan":
        available_months = sorted(
            df["datetime"]
            .dt.to_period("M")
            .dropna()
            .unique()
            .astype(str)
            .tolist()
        )

        if not available_months:
            return df.iloc[0:0].copy()

        selected_month = st.selectbox(
            "Pilih bulan",
            available_months
        )

        filtered = df[
            df["datetime"].dt.to_period("M").astype(str)
            == selected_month
        ].copy()

    else:
        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "Tanggal mulai",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )

        with col2:
            end_date = st.date_input(
                "Tanggal akhir",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )

        if start_date > end_date:
            st.error(
                "Tanggal mulai tidak boleh melewati tanggal akhir."
            )
            return df.iloc[0:0].copy()

        filtered = df[
            (df["datetime"].dt.date >= start_date)
            & (df["datetime"].dt.date <= end_date)
        ].copy()

    return filtered


# ============================================================
# 10. EXPORT EXCEL
# ============================================================

def create_excel_export(
    ranking,
    period_df,
    monthly_summary,
    review_df
):
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        ranking.to_excel(
            writer,
            sheet_name="Ranking Periode",
            index=False
        )

        period_df.to_excel(
            writer,
            sheet_name="Pesan Periode",
            index=False
        )

        monthly_summary.to_excel(
            writer,
            sheet_name="Perbandingan Bulanan",
            index=False
        )

        review_df.to_excel(
            writer,
            sheet_name="Spam Review",
            index=False
        )

    output.seek(0)

    return output.getvalue()


# ============================================================
# 11. UPLOAD FILE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload file ekspor WhatsApp (.txt)",
    type=["txt"]
)

if uploaded_file is None:
    st.info(
        "Upload file ekspor chat WhatsApp dalam format .txt "
        "untuk mulai menganalisis komunitas."
    )

    st.markdown(
        """
        **Cara mendapatkan file WhatsApp:**
        1. Buka grup WhatsApp.
        2. Buka menu grup.
        3. Pilih Ekspor Chat.
        4. Pilih Tanpa Media.
        5. Upload file `.txt` ke aplikasi ini.
        """
    )

    st.stop()


# ============================================================
# 12. PROSES DATA
# ============================================================

with st.spinner("Membaca dan menganalisis percakapan..."):

    raw_df = parse_whatsapp_chat(uploaded_file)

if raw_df.empty:
    st.error(
        "Tidak ada pesan yang berhasil dibaca. "
        "Pastikan file berasal dari ekspor chat WhatsApp "
        "dan menggunakan format tanggal yang didukung."
    )
    st.stop()

classified_df = classify_messages(raw_df)

# Perbandingan bulanan menggunakan seluruh file.
all_data_df = apply_moderate_spam_rules(
    classified_df.copy()
)

monthly_summary = build_monthly_summary(
    all_data_df
)

# Filter periode untuk ranking dan detail.
period_df = filter_period(classified_df)

if period_df.empty:
    st.warning("Tidak ada pesan pada periode yang dipilih.")
    st.stop()

period_df = apply_moderate_spam_rules(
    period_df.copy()
)

ranking = build_member_ranking(period_df)

if ranking.empty:
    st.warning(
        "Tidak ada pesan anggota yang dapat dianalisis "
        "pada periode ini."
    )
    st.stop()

review_df = period_df[
    period_df["needs_review"]
    | period_df["spam_penalty_flag"]
].copy()


# ============================================================
# 13. KPI DASHBOARD
# ============================================================

st.divider()
st.subheader("📊 Community Overview")

total_messages = len(
    period_df[
        ~period_df["is_system"]
        & ~period_df["is_empty"]
    ]
)

total_members = period_df[
    ~period_df["is_system"]
    & ~period_df["is_empty"]
]["sender"].nunique()

total_active_days = period_df[
    ~period_df["is_system"]
    & ~period_df["is_empty"]
]["datetime"].dt.date.nunique()

total_review = int(
    period_df["needs_review"].sum()
)

total_strong_spam = int(
    period_df["spam_penalty_flag"].sum()
)

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Total Pesan Anggota",
    f"{total_messages:,}"
)

k2.metric(
    "Anggota Aktif",
    f"{total_members:,}"
)

k3.metric(
    "Hari Aktif",
    f"{total_active_days:,}"
)

k4.metric(
    "Perlu Ditinjau",
    f"{total_review:,}"
)

k5.metric(
    "Indikasi Spam Kuat",
    f"{total_strong_spam:,}"
)


# ============================================================
# 14. RANKING AKTIVITAS
# ============================================================

st.divider()
st.subheader("🏃 Activity Ranking")

st.caption(
    "Mengukur volume pesan dan konsistensi keaktifan. "
    "Skor relatif terhadap anggota lain pada periode terpilih."
)

activity_ranking = ranking.sort_values(
    ["activity_score", "total_messages"],
    ascending=False
).copy()

activity_columns = [
    "activity_rank",
    "member",
    "total_messages",
    "active_days",
    "activity_volume_score",
    "consistency_score",
    "activity_score"
]

st.dataframe(
    activity_ranking[activity_columns].rename(
        columns={
            "activity_rank": "Peringkat",
            "member": "Nama Anggota",
            "total_messages": "Total Pesan",
            "active_days": "Hari Aktif",
            "activity_volume_score": "Skor Volume",
            "consistency_score": "Skor Konsistensi",
            "activity_score": "Activity Score"
        }
    ),
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    activity_ranking.set_index("member")[
        "activity_score"
    ].head(15)
)


# ============================================================
# 15. RANKING KONTRIBUSI
# ============================================================

st.divider()
st.subheader("🌟 Contribution Ranking")

st.caption(
    "Menggunakan indikator pesan substantif, sinyal membantu, "
    "konsistensi, dan penalti untuk pola spam kuat."
)

contribution_ranking = ranking.sort_values(
    ["contribution_score", "substantive_messages"],
    ascending=False
).copy()

contribution_columns = [
    "contribution_rank",
    "member",
    "total_messages",
    "substantive_messages",
    "helpful_signals",
    "substantive_rate_pct",
    "helpful_rate_pct",
    "spam_penalty_points",
    "contribution_score"
]

st.dataframe(
    contribution_ranking[contribution_columns].rename(
        columns={
            "contribution_rank": "Peringkat",
            "member": "Nama Anggota",
            "total_messages": "Total Pesan",
            "substantive_messages": "Pesan Substantif",
            "helpful_signals": "Sinyal Membantu",
            "substantive_rate_pct": "Pesan Substantif (%)",
            "helpful_rate_pct": "Sinyal Membantu (%)",
            "spam_penalty_points": "Penalti Spam",
            "contribution_score": "Contribution Score"
        }
    ),
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    contribution_ranking.set_index("member")[
        "contribution_score"
    ].head(15)
)


# ============================================================
# 16. DETAIL AKTIVITAS ANGGOTA
# ============================================================

st.divider()
st.subheader("🔎 Detail Anggota")

member_list = sorted(
    ranking["member"].dropna().unique().tolist()
)

selected_member = st.selectbox(
    "Pilih anggota untuk melihat detail",
    member_list
)

member_messages = period_df[
    period_df["sender"] == selected_member
].copy()

member_messages = member_messages[
    ~member_messages["is_system"]
    & ~member_messages["is_empty"]
]

member_row = ranking[
    ranking["member"] == selected_member
]

if not member_row.empty:

    row = member_row.iloc[0]

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Total Pesan",
        int(row["total_messages"])
    )

    d2.metric(
        "Hari Aktif",
        int(row["active_days"])
    )

    d3.metric(
        "Activity Score",
        round(row["activity_score"], 2)
    )

    d4.metric(
        "Contribution Score",
        round(row["contribution_score"], 2)
    )

st.dataframe(
    member_messages[
        [
            "datetime",
            "message",
            "is_substantive",
            "has_helpful_signal",
            "is_duplicate",
            "is_promo_suspect",
            "spam_penalty_flag"
        ]
    ].rename(
        columns={
            "datetime": "Waktu",
            "message": "Pesan",
            "is_substantive": "Substantif",
            "has_helpful_signal": "Sinyal Membantu",
            "is_duplicate": "Pesan Duplikat",
            "is_promo_suspect": "Indikasi Promosi",
            "spam_penalty_flag": "Pola Spam Kuat"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 17. SPAM REVIEW
# ============================================================

st.divider()
st.subheader("🛡️ Spam Review")

st.write(
    "Bagian ini menampilkan pesan yang perlu ditinjau. "
    "Pesan yang mengandung tautan, promosi, atau duplikat "
    "tidak otomatis dianggap spam."
)

if review_df.empty:

    st.success(
        "Tidak ditemukan pesan yang perlu ditinjau "
        "berdasarkan aturan saat ini."
    )

else:

    review_filter = st.radio(
        "Jenis pemeriksaan",
        [
            "Semua Perlu Ditinjau",
            "Duplikat",
            "Promosi atau Tautan",
            "Indikasi Spam Kuat"
        ],
        horizontal=True
    )

    if review_filter == "Duplikat":

        display_review = review_df[
            review_df["is_duplicate"]
        ].copy()

    elif review_filter == "Promosi atau Tautan":

        display_review = review_df[
            review_df["is_promo_suspect"]
        ].copy()

    elif review_filter == "Indikasi Spam Kuat":

        display_review = review_df[
            review_df["spam_penalty_flag"]
        ].copy()

    else:

        display_review = review_df.copy()

    review_columns = [
        "datetime",
        "sender",
        "message",
        "is_duplicate",
        "is_promo_suspect",
        "same_message_count",
        "promo_suspect_count_by_sender",
        "spam_status",
        "strong_spam_reason"
    ]

    st.dataframe(
        display_review[review_columns].rename(
            columns={
                "datetime": "Waktu",
                "sender": "Nama Anggota",
                "message": "Isi Pesan",
                "is_duplicate": "Pesan Duplikat",
                "is_promo_suspect": "Indikasi Promosi",
                "same_message_count": "Jumlah Pesan Identik",
                "promo_suspect_count_by_sender":
                    "Jumlah Indikasi Promosi Anggota",
                "spam_status": "Status",
                "strong_spam_reason": "Alasan Indikasi Kuat"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 18. PERBANDINGAN BULANAN
# ============================================================

st.divider()
st.subheader("📅 Monthly Community Comparison")

st.caption(
    "Bagian ini menggunakan seluruh file yang di-upload, "
    "bukan hanya periode yang dipilih untuk ranking."
)

if monthly_summary.empty:

    st.info("Belum ada data yang dapat dibandingkan.")

else:

    st.dataframe(
        monthly_summary.rename(
            columns={
                "bulan": "Bulan",
                "total_pesan": "Total Pesan",
                "jumlah_member": "Jumlah Anggota",
                "hari_aktif_grup": "Hari Aktif Grup",
                "pesan_perlu_ditinjau": "Perlu Ditinjau",
                "indikasi_spam_kuat": "Indikasi Spam Kuat",
                "pesan_substantif": "Pesan Substantif",
                "rata_pesan_per_member": "Rata-rata Pesan per Anggota"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    chart_data = monthly_summary.set_index("bulan")[
        [
            "total_pesan",
            "pesan_substantif",
            "pesan_perlu_ditinjau"
        ]
    ]

    st.line_chart(chart_data)


# ============================================================
# 19. DOWNLOAD EXCEL
# ============================================================

st.divider()
st.subheader("📥 Export Data")

st.write(
    "Unduh hasil analisis untuk dokumentasi, evaluasi "
    "komunitas, atau analisis lanjutan."
)

excel_bytes = create_excel_export(
    ranking=ranking,
    period_df=period_df,
    monthly_summary=monthly_summary,
    review_df=review_df
)

st.download_button(
    label="Download Hasil Analisis (Excel)",
    data=excel_bytes,
    file_name="HPN_Community_Analysis.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    )
)


# ============================================================
# 20. CATATAN METODOLOGI
# ============================================================

with st.expander("ℹ️ Cara membaca hasil analisis"):

    st.markdown(
        """
        **Activity Score**

        Mengukur volume pesan dan konsistensi hari aktif.
        Skor tinggi menunjukkan aktivitas relatif tinggi
        dibandingkan anggota lain pada periode yang sama.

        **Contribution Score**

        Menggunakan indikator pesan substantif, sinyal membantu,
        konsistensi, dan penalti untuk pola spam kuat.

        **Pesan substantif**

        Diidentifikasi melalui panjang pesan atau penggunaan
        tanda tanya. Pesan panjang belum tentu berkualitas.

        **Sinyal membantu**

        Berdasarkan kata atau frasa tertentu. Sistem belum
        memahami konteks secara penuh, sehingga sinyal ini
        bukan bukti bahwa pesan benar-benar membantu.

        **Spam Review**

        Duplikat dan promosi atau tautan mencurigakan
        ditampilkan untuk pemeriksaan. Satu pesan duplikat
        atau satu tautan tidak otomatis mengurangi skor.

        **Indikasi spam kuat**

        Diberikan ketika pesan identik muncul minimal tiga kali,
        atau anggota mengirim minimal tiga pesan yang terindikasi
        promosi atau tautan mencurigakan dalam periode analisis.

        **Penting**

        Hasil merupakan indikator awal untuk membantu moderator.
        Periksa konteks sebelum mengambil tindakan terhadap anggota.
        """
    )

st.caption(
    "HPN Community Analyzer | Internal Community Intelligence Tool"
)
