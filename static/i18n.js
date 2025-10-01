;(function () {
  const he = {
    title: "תמלול בעזרת ivrit.ai",
    themeToggle: "החלף מצב תאורה",
    settings: "הגדרות",
    language: "שפה",
    languageHebrew: "עברית",
    languageYiddish: "ייִדיש",
    languageEnglish: "אנגלית",
    dropPrompt: "גרור קובץ לכאן או לחץ לבחירת קובץ",
    transcribe: "תמלל",
    copyText: "העתק את הטקסט",
    downloads: "הורדות",
    downloadAs: "הורד כ...",
    wordDocx: "Word DOCX",
    contactUs: "רוצים ליצור קשר? אנחנו כאן",
    donateAsk: "רוצים לתרום לנו?",
    privacyTos: "מדיניות פרטיות ותנאי שימוש",
    settingsHeader: "הגדרות",
    loading: "טוען...",
    displaySettings: "הגדרות תצוגה",
    timestampsLabel: "קודי זמן:",
    timestampsNone: "ללא קודי זמן",
    timestampsNoneHelp: "רק הטקסט ללא זמנים",
    timestampsSegments: "כולל קודי זמן",
    timestampsSegmentsHelp: "זמני התחלה וסיום, לפי פורמט הטקסט",
    diarizationLabel: "הפרדת דוברים:",
    diarizationEnabled: "הצג דוברים",
    diarizationEnabledHelp: "הצג תוויות דוברים ושורות חדשות עם החלפת דובר",
    diarizationDisabled: "הסתר דוברים",
    diarizationDisabledHelp: "הצג רק טקסט ללא תוויות דוברים",
    save: "שמור",
    cancel: "ביטול",
    clear: "נקה הגדרות",
    fileSelected: "קובץ נבחר",
    fileSize: "גודל קובץ",
    removeFile: "הסר קובץ",
    fileRemovedSuccess: "קובץ הוסר בהצלחה",
    fileSelectedSuccess: "קובץ נבחר בהצלחה",
    fileTooLarge: "הקובץ גדול מדי. הגודל המקסימלי המותר הוא {size}",
    uploading: "מעלה את הקובץ...",
    uploadingProgress: "מעלה את הקובץ... {progress}%",
    uploadedStarting: "קובץ הועלה. מתחיל בתמלול...",
    uploadError: "שגיאה בהעלאת הקובץ. אנא נסה שוב.",
    queuePosition: "בתור במקום {position}. זמן משוער להתחלה: {eta}",
    jobTypePrefix: "סוג עבודה: {jobType}",
    jobTypeShort: "קצרה",
    jobTypeLong: "ארוכה",
    jobTypePrivate: "פרטית",
    waitingMsg:
      "בזמן שאתם ממתינים, אנו מזכירים ש-ivrit.ai הוא פרויקט ללא מטרות רווח.\nכל השירותים שלנו, כולל שירות התמלול בו אתם משתמשים כרגע, ניתנים בחינם.\n\nנודה אם תוכלו לתרום בכדי שנוכל להנגיש את השירות ליותר משתמשים.\n\nניתן לתרום דרך לינק לפטראון בתחתית המסך.\n\nתודה!",
    transcribing: "מתמלל... {progress}%",
    statusError: "אירעה שגיאה בבדיקת סטטוס העבודה",
    verifyChecking: "בודק...",
    settingsSaved: "הגדרות נשמרו בהצלחה",
    settingsCleared: "כל ההגדרות נמחקו בהצלחה",
    balanceLoadingError: "שגיאה בטעינה",
    errorReportTemplate:
      "אם השגיאה נמשכת, אנא שלחו דיווח לכתובת info@ivrit.ai עם הפרטים הבאים:\n- כתובת האימייל שלכם\n- זמן השגיאה: {time}\n- פרטי השגיאה: {details}",
    transcriptFooter: "תומלל באמצעות שירות התמלול של ivrit.ai",
    copySuccess: "הטקסט הועתק ללוח בהצלחה",
    docxLibMissing: "ספריית יצירת המסמך לא נטענה. בדקו את החיבור ונסו שוב.",
    docxCreateError: "שגיאה ביצירת קובץ Word. אנא נסה שוב.",
    exportError: "שגיאה בייצוא נתוני התמלול",
    exitWarning:
      "התמלול עדיין פעיל. יציאה מהדף תפסיק את התמלול. האם אתה בטוח שברצונך לצאת?",
    transcriptHeaderDocx: "תמלול",
    speaker: "דובר {num}",
    speakerUnknown: "דובר לא מזוהה",
    errorPrefix: "שגיאה",
    unitBytes: "Bytes",
    unitKB: "KB",
    unitMB: "MB",
    unitGB: "GB",
    textFormatSegmentsTitle: "פורמט הטקסט: פסקאות נפרדות",
    textFormatContinuousTitle: "פורמט הטקסט: טקסט רציף",
    filesSelectedCount: "{count} קבצים נבחרו",
    filesSelectedSuccess: "{count} קבצים נבחרו בהצלחה",
    batchStarting: "מתחיל עיבוד אצווה של {count} קבצים...",
    batchUploading: "מעלה {index}/{total}: {filename}",
    batchQueued:
      "בתור ({index}/{total}) {filename} · מקום {position} · זמן משוער: {eta}",
    batchTranscribing: "מתמלל ({index}/{total}) {filename} · {progress}%",
    batchDone: "האצווה הושלמה",
    docxDownloaded: "קובץ DOCX ירד: {filename}",
  }

  const yi = {
    title: "טראַנסקריפּציע דורך ivrit.ai",
    themeToggle: "באַשטייערן ליכט/טונקל מאָדע",
    settings: "איינשטעלונגען",
    language: "שפּראַך",
    languageHebrew: "עברית",
    languageYiddish: "ייִדיש",
    languageEnglish: "ענגליש",
    dropPrompt: "ציע לאָז דאָ אַ טעקע אָדער דריקט צו אויסקלייבן",
    transcribe: "טראַנסקריבער",
    copyText: "קאָפּירן דעם טעקסט",
    downloads: "אַראָפלאָדן",
    downloadAs: "אַראָפלאָדן ווי...",
    wordDocx: "Word DOCX",
    contactUs: "ווילט איר זיך קאָנטאַקטירן? מיר זײַנען דאָ",
    donateAsk: "ווילט איר אונדז שטיצן?",
    privacyTos: "פּריוואַטקייט פּאָליסי און תנאי־ניצול",
    settingsHeader: "איינשטעלונגען",
    loading: "לאָדט...",
    displaySettings: "וויזועל־איינשטעלונגען",
    timestampsLabel: "צײַט־קאָדן:",
    timestampsNone: "אָן צײַט־קאָדן",
    timestampsNoneHelp: "בלויז דער טעקסט אָן צײַטן",
    timestampsSegments: "מיט צײַט־קאָדן",
    timestampsSegmentsHelp: "אָנפֿאַנג און סוף, לויט פֿאָרמאַט",
    diarizationLabel: "אָפּטיילונג פֿון רעדנדיקע:",
    diarizationEnabled: "ווײַזן רעדנדיקע",
    diarizationEnabledHelp: "ווײַזן טאַגלען און נײַע שורות בײַ טויש",
    diarizationDisabled: "באַהאַלטן רעדנדיקע",
    diarizationDisabledHelp: "ווײַזן בלויז טעקסט אָן טאַגלען",
    save: "אויפֿהיטן",
    cancel: "אַנולירן",
    clear: "באַוואַרפֿן איינשטעלונגען",
    fileSelected: "דערקליבענע טעקע",
    fileSize: "טעקע־גרייס",
    removeFile: "אויסמעקן טעקע",
    fileRemovedSuccess: "טעקע איז געלאָשן געוואָרן מיט הצלחה",
    fileSelectedSuccess: "טעקע איז געראָטן אויסגעקליבן",
    fileTooLarge: "די טעקע איז צו גרויַס. מעגלעכער העכסטער איז {size}",
    uploading: "אַרויפֿלאָדט די טעקע...",
    uploadingProgress: "אַרויפֿלאָדט די טעקע... {progress}%",
    uploadedStarting: "טעקע אַרויפֿגעלאָדן. הייבט אָן טראַנסקריפּציע...",
    uploadError: "א טעות בעת אַרויפֿלאָדן. ביטע פּרוּווט נאָך אַ מאָל.",
    queuePosition:
      "אין דער ריי אָרט {position}. אַפּראָקס. אָנהייב־צײַט: {eta}",
    jobTypePrefix: "אַרבעט־סאָרט: {jobType}",
    jobTypeShort: "קורץ",
    jobTypeLong: "לאַנג",
    jobTypePrivate: "פּריוואַט",
    waitingMsg:
      "ווען איר וואַרט, געדענקט אַז ivrit.ai איז אַ נאַן־פּראָפֿיט.\nאַלע אונדזערע דינען, אַרײַנגערעכנט דעם טראַנסקריפּציע־דינסט, זײַנען בחינם.\n\nביטע שטיצט אונדז כּדי צוצוגעבן דעם דינסט פאר מער ניצערס.\n\nמען קען שטיצן דורך פּאַטרעאָן לינק אונטן.\n\nאַ דאַנק!",
    transcribing: "טראַנסקריבירט... {progress}%",
    statusError: "א טעות בײַם קאָנטראָלירן אַרבעט־סטאַטוס",
    verifyChecking: "באַשטעטיקט...",
    settingsSaved: "איינשטעלונגען געראַטעוועט",
    settingsCleared: "אַלע איינשטעלונגען ווערן געלייגט צוריק",
    balanceLoadingError: "טעות בײַם לאָדן",
    errorReportTemplate:
      "אויב דער טעות בלײַבט, ביטע שיקט אַ רעפּאָרט צו info@ivrit.ai מיט די דעטאַלן:\n- אייער אימעיל־אַדרעס\n- צײַט פֿון טעות: {time}\n- דעטאַלן: {details}",
    transcriptFooter: "טראַנסקריבירט דורך ivrit.ai'ס דינסט",
    copySuccess: "טעקסט איז קאַפּירט געוואָרן",
    docxLibMissing:
      "די דאָקומענט־ביבליאָטעק איז נישט געלאָדן. פּרוּווט ווידער.",
    docxCreateError: "טעות בײַם שאַפֿן Word טעקע. פּרוּווט ווידער.",
    exportError: "טעות בײַם אַן־פיר פֿון טראַנסקריפּציע־דאַטן",
    exitWarning:
      "טראַנסקריפּציע לויפֿט נאָך. אַרויסגיין וועט אָפּשטעלן. זענט איר זיכער?",
    transcriptHeaderDocx: "טראַנסקריפּציע",
    speaker: "רעדנדיקער {num}",
    speakerUnknown: "ניט־ידענטיפֿיצירטער רעדנדיקער",
    errorPrefix: "טעות",
    unitBytes: "Bytes",
    unitKB: "KB",
    unitMB: "MB",
    unitGB: "GB",
    textFormatSegmentsTitle: "טעקסט־פֿאָרמאַט: באַזונדערע פּאַראַגראַפֿן",
    textFormatContinuousTitle: "טעקסט־פֿאָרמאַט: קאַנטיניואַלע שורה",
    filesSelectedCount: "{count} טעקעס אויסגעקליבן",
    filesSelectedSuccess: "{count} טעקעס געראָטן אויסגעקליבן",
    batchStarting: "הייבט אָן אַ באַטש פֿון {count} טעקעס...",
    batchUploading: "אַרויפֿלאָדט {index}/{total}: {filename}",
    batchQueued:
      "אין ריי ({index}/{total}) {filename} · אָרט {position} · אַפּראָקס. {eta}",
    batchTranscribing:
      "טראַנסקריבירט ({index}/{total}) {filename} · {progress}%",
    batchDone: "באַטש פאַרענדיקט",
    docxDownloaded: "DOCX אַראָפּלאָדן: {filename}",
  }

  const en = {
    title: "Transcription by ivrit.ai",
    themeToggle: "Toggle light/dark mode",
    settings: "Settings",
    language: "Language",
    languageHebrew: "Hebrew",
    languageYiddish: "Yiddish",
    languageEnglish: "English",
    dropPrompt: "Drag a file here or click to choose",
    transcribe: "Transcribe",
    copyText: "Copy text",
    downloads: "Downloads",
    downloadAs: "Download as...",
    wordDocx: "Word DOCX",
    contactUs: "Want to contact us? We're here",
    donateAsk: "Want to support us?",
    privacyTos: "Privacy Policy and Terms of Use",
    settingsHeader: "Settings",
    loading: "Loading...",
    displaySettings: "Display settings",
    timestampsLabel: "Timestamps:",
    timestampsNone: "No timestamps",
    timestampsNoneHelp: "Only text without times",
    timestampsSegments: "Include timestamps",
    timestampsSegmentsHelp: "Start and end times, per text format",
    diarizationLabel: "Speaker separation:",
    diarizationEnabled: "Show speakers",
    diarizationEnabledHelp: "Show speaker labels and break lines on change",
    diarizationDisabled: "Hide speakers",
    diarizationDisabledHelp: "Show only text without speaker labels",
    save: "Save",
    cancel: "Cancel",
    clear: "Clear settings",
    fileSelected: "File selected",
    fileSize: "File size",
    removeFile: "Remove file",
    fileRemovedSuccess: "File removed successfully",
    fileSelectedSuccess: "File selected successfully",
    fileTooLarge: "File too large. Maximum allowed is {size}",
    uploading: "Uploading file...",
    uploadingProgress: "Uploading file... {progress}%",
    uploadedStarting: "File uploaded. Starting transcription...",
    uploadError: "Upload error. Please try again.",
    queuePosition: "In queue position {position}. Estimated start: {eta}",
    jobTypePrefix: "Job type: {jobType}",
    jobTypeShort: "short",
    jobTypeLong: "long",
    jobTypePrivate: "private",
    waitingMsg:
      "While you wait, note ivrit.ai is a non-profit. All our services, including this transcription service, are free.\n\nPlease consider supporting us so we can make this service available to more users.\n\nYou can support via the Patreon link at the bottom of the page.\n\nThank you!",
    transcribing: "Transcribing... {progress}%",
    statusError: "An error occurred while checking job status",
    verifyChecking: "Checking...",
    settingsSaved: "Settings saved",
    settingsCleared: "All settings were cleared",
    balanceLoadingError: "Loading error",
    errorReportTemplate:
      "If the error persists, please email info@ivrit.ai with:\n- Your email address\n- Error time: {time}\n- Error details: {details}",
    transcriptFooter: "Transcribed using ivrit.ai's transcription service",
    copySuccess: "Text copied to clipboard",
    docxLibMissing:
      "Document library not loaded. Check connection and try again.",
    docxCreateError: "Error creating Word file. Please try again.",
    exportError: "Error exporting transcription data",
    exitWarning:
      "Transcription is still running. Leaving will stop it. Are you sure you want to leave?",
    transcriptHeaderDocx: "Transcription",
    speaker: "Speaker {num}",
    speakerUnknown: "Unknown speaker",
    errorPrefix: "Error",
    unitBytes: "Bytes",
    unitKB: "KB",
    unitMB: "MB",
    unitGB: "GB",
    textFormatSegmentsTitle: "Text format: separate paragraphs",
    textFormatContinuousTitle: "Text format: continuous text",
    filesSelectedCount: "{count} files selected",
    filesSelectedSuccess: "{count} files selected successfully",
    batchStarting: "Starting batch of {count} files...",
    batchUploading: "Uploading {index}/{total}: {filename}",
    batchQueued:
      "Queued ({index}/{total}) {filename} · position {position} · ETA {eta}",
    batchTranscribing:
      "Transcribing ({index}/{total}) {filename} · {progress}%",
    batchDone: "Batch complete",
    docxDownloaded: "DOCX downloaded: {filename}",
  }

  const resources = { he, yi, en }

  function interpolate(template, vars) {
    return String(template).replace(/\{(.*?)\}/g, (_, k) =>
      Object.prototype.hasOwnProperty.call(vars || {}, k) ? vars[k] : `{${k}}`
    )
  }

  const I18N = {
    current: "he",
    setLanguage(lang) {
      this.current = resources[lang] ? lang : "he"
      try {
        localStorage.setItem("ui_lang", this.current)
      } catch {}
      document.documentElement.setAttribute("lang", this.current)
      const rtlLangs = ["he", "yi"]
      document.documentElement.setAttribute(
        "dir",
        rtlLangs.includes(this.current) ? "rtl" : "ltr"
      )
    },
    t(key, vars) {
      const langTable = resources[this.current] || resources["he"]
      const base =
        (langTable && langTable[key]) ||
        (resources["he"] && resources["he"][key]) ||
        key
      return interpolate(base, vars)
    },
    apply() {
      const langTable = resources[this.current] || resources["he"]
      try {
        document.title = this.t("title")
      } catch {}
      document.querySelectorAll("[data-i18n]").forEach((el) => {
        const key = el.getAttribute("data-i18n")
        if (!key) return
        const val = this.t(key)
        if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") {
          el.setAttribute("placeholder", val)
        } else {
          el.textContent = val
        }
      })
      document.querySelectorAll("[data-i18n-title]").forEach((el) => {
        const key = el.getAttribute("data-i18n-title")
        el.setAttribute("title", this.t(key))
        el.setAttribute("aria-label", this.t(key))
      })
    },
  }

  window.I18N = I18N
  // Initialize language from storage or default
  try {
    const saved = localStorage.getItem("ui_lang") || "he"
    I18N.setLanguage(saved)
  } catch {
    I18N.setLanguage("he")
  }
})()
