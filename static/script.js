const loader = document.getElementById("loader");
const video = document.getElementById("video");
const emotionText = document.getElementById("emotion");
const explanationText = document.getElementById("explanation");
const recDiv = document.getElementById("recommendations");
const timeText = document.getElementById("time-bias");
const confidenceText = document.getElementById("confidence");


/* ================= CAMERA ================= */

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.play();
    })
    .catch(err => console.log("Camera error:", err));


/* ================= FACE DETECTION ================= */

setInterval(async () => {
    loader.classList.remove("hidden");
    if (video.videoWidth === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 240;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, 320, 240);

    const imgData = canvas.toDataURL("image/jpeg");

    const res = await fetch("/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imgData })
    });

    const data = await res.json();
    loader.classList.add("hidden");

    // Update emotion info
    emotionText.innerText = data.emotion;
    confidenceText.innerText = data.confidence + "%";
    timeText.innerText = data.time_bias;
    explanationText.innerText = data.explanation;


    /* ================= RECOMMENDATIONS ================= */

    recDiv.innerHTML = "";

    if (!data.recommendations || data.recommendations.length === 0) {
        recDiv.innerHTML = "<p>No recommendations available</p>";
        return;
    }

    data.recommendations.forEach(item => {

        const card = document.createElement("div");
        card.className = "rec-card";

        card.innerHTML = `
            <img src="${item.image}"
                 onerror="this.src='/static/default.png'" />

            <h4>${item.title}</h4>

            <a href="${item.url}" target="_blank">
                ${item.button}
            </a>
        `;

        recDiv.appendChild(card);
    });

}, 2000);
