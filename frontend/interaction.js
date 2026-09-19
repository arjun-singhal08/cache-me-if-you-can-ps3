/* ============================================================
   CACHE ME IF YOU CAN — SHM CONTROL SYSTEM
   Interaction Layer
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {

    /* ---------------------------------------------------------
       PAGE ENTRANCE
       --------------------------------------------------------- */

    const animatedElements = document.querySelectorAll(
        ".card, .metric-card, .panel, .upload-zone, .chart-container, .damage-display"
    );

    animatedElements.forEach((element, index) => {
        element.style.opacity = "0";
        element.style.transform = "translateY(12px)";

        setTimeout(() => {
            element.style.opacity = "1";
            element.style.transform = "translateY(0)";
        }, Math.min(index * 65, 500));
    });


    /* ---------------------------------------------------------
       BUTTON RIPPLE
       --------------------------------------------------------- */

    document.addEventListener("click", event => {

        const button = event.target.closest("button");

        if (!button) return;

        const ripple = document.createElement("span");
        const rect = button.getBoundingClientRect();

        const size = Math.max(
            rect.width,
            rect.height
        );

        ripple.style.width = `${size}px`;
        ripple.style.height = `${size}px`;

        ripple.style.left =
            `${event.clientX - rect.left - size / 2}px`;

        ripple.style.top =
            `${event.clientY - rect.top - size / 2}px`;

        ripple.classList.add("button-ripple");

        button.appendChild(ripple);

        setTimeout(() => {
            ripple.remove();
        }, 600);

    });


    /* ---------------------------------------------------------
       DRAG/DROP VISUAL STATE
       --------------------------------------------------------- */

    const uploadZones = document.querySelectorAll(
        ".upload-zone"
    );

    uploadZones.forEach(zone => {

        ["dragenter", "dragover"].forEach(eventName => {

            zone.addEventListener(eventName, event => {

                event.preventDefault();

                zone.classList.add(
                    "drag-active"
                );

            });

        });


        ["dragleave", "drop"].forEach(eventName => {

            zone.addEventListener(eventName, () => {

                zone.classList.remove(
                    "drag-active"
                );

            });

        });

    });


    /* ---------------------------------------------------------
       ANIMATED NUMBERS

       Example:
       animateNumber(
           document.querySelector("#damageValue"),
           0.347,
           3
       );
       --------------------------------------------------------- */

    window.animateNumber = function (
        element,
        target,
        decimals = 2,
        duration = 900
    ) {

        if (!element) return;

        target = Number(target);

        const start =
            Number(element.dataset.currentValue) || 0;

        const startTime =
            performance.now();

        function update(time) {

            const elapsed =
                time - startTime;

            const progress =
                Math.min(
                    elapsed / duration,
                    1
                );

            const eased =
                1 - Math.pow(
                    1 - progress,
                    3
                );

            const value =
                start +
                (target - start) * eased;

            element.textContent =
                value.toFixed(decimals);

            if (progress < 1) {

                requestAnimationFrame(
                    update
                );

            } else {

                element.dataset.currentValue =
                    target;

            }

        }

        requestAnimationFrame(update);
    };


    /* ---------------------------------------------------------
       HEALTH STATUS

       Safe:       D < 0.20
       Moderate:   0.20 <= D <= 0.55
       Critical:   D > 0.55
       --------------------------------------------------------- */

    window.updateHealthStatus = function (damage) {

        damage = Number(damage);

        const badge =
            document.querySelector(
                "#healthStatus"
            );

        if (!badge) return;

        badge.classList.remove(
            "status-safe",
            "status-warning",
            "status-critical",
            "status-pop"
        );

        let status;
        let cssClass;

        if (damage < 0.20) {

            status = "SAFE";
            cssClass = "status-safe";

        }

        else if (damage <= 0.55) {

            status = "MODERATE WARNING";
            cssClass = "status-warning";

        }

        else {

            status = "CRITICAL FATIGUE";
            cssClass = "status-critical";

        }

        badge.classList.add(cssClass);

        badge.innerHTML = `
            <span class="status-light"></span>
            ${status}
        `;

        void badge.offsetWidth;

        badge.classList.add(
            "status-pop"
        );

    };


    /* ---------------------------------------------------------
       DAMAGE BAR / GAUGE
       --------------------------------------------------------- */

    window.animateDamageGauge = function (damage) {

        damage = Math.max(
            0,
            Math.min(1, Number(damage))
        );

        const gauge =
            document.querySelector(
                "#damageGauge"
            );

        if (!gauge) return;

        const fill =
            gauge.querySelector(
                ".damage-gauge-fill"
            );

        const marker =
            gauge.querySelector(
                ".damage-marker"
            );

        if (fill) {

            requestAnimationFrame(() => {

                fill.style.width =
                    `${damage * 100}%`;

            });

        }

        if (marker) {

            marker.style.left =
                `${damage * 100}%`;

        }

    };


    /* ---------------------------------------------------------
       TOAST NOTIFICATIONS

       showToast(
           "Analysis complete",
           "success"
       );
       --------------------------------------------------------- */

    window.showToast = function (
        message,
        type = "info"
    ) {

        let container =
            document.querySelector(
                ".toast-container"
            );

        if (!container) {

            container =
                document.createElement("div");

            container.className =
                "toast-container";

            document.body.appendChild(
                container
            );

        }


        const toast =
            document.createElement("div");

        toast.className =
            `toast toast-${type}`;


        const symbols = {

            success: "✓",
            error: "×",
            warning: "!",
            info: "i"

        };


        toast.innerHTML = `

            <div class="toast-symbol">
                ${symbols[type] || "i"}
            </div>

            <div class="toast-copy">

                <span class="toast-title">
                    ${
                        type === "success"
                            ? "SYSTEM"
                            : type === "error"
                            ? "ERROR"
                            : type === "warning"
                            ? "WARNING"
                            : "NOTICE"
                    }
                </span>

                <span>
                    ${message}
                </span>

            </div>

        `;


        container.appendChild(toast);


        requestAnimationFrame(() => {

            toast.classList.add(
                "toast-visible"
            );

        });


        setTimeout(() => {

            toast.classList.remove(
                "toast-visible"
            );

            setTimeout(() => {

                toast.remove();

            }, 300);

        }, 3200);

    };


    /* ---------------------------------------------------------
       MODEL LOADING SCREEN

       showModelLoader();
       hideModelLoader();
       --------------------------------------------------------- */

    window.showModelLoader = function (
        message = "Processing stress signal..."
    ) {

        let loader =
            document.querySelector(
                "#modelLoader"
            );


        if (!loader) {

            loader =
                document.createElement("div");

            loader.id =
                "modelLoader";

            loader.className =
                "model-loader-overlay";


            loader.innerHTML = `

                <div class="model-loader-card">

                    <div class="loader-top">

                        <span>
                            SHM ANALYSIS
                        </span>

                        <span class="loader-code">
                            PROC / 001
                        </span>

                    </div>


                    <div class="rail-animation">

                        <div class="rail-line"></div>

                        <div class="rail-scan"></div>

                    </div>


                    <h3>
                        ANALYSING STRUCTURAL SIGNAL
                    </h3>


                    <p id="modelLoaderText">
                        ${message}
                    </p>


                    <div class="analysis-progress">

                        <div
                            class="analysis-progress-bar">
                        </div>

                    </div>


                    <div class="loader-footer">

                        <span>
                            FATIGUE MODEL
                        </span>

                        <span>
                            RUNNING
                        </span>

                    </div>

                </div>

            `;


            document.body.appendChild(
                loader
            );

        }


        const loaderText =
            loader.querySelector(
                "#modelLoaderText"
            );


        if (loaderText) {

            loaderText.textContent =
                message;

        }


        requestAnimationFrame(() => {

            loader.classList.add(
                "active"
            );

        });

    };


    window.hideModelLoader = function () {

        const loader =
            document.querySelector(
                "#modelLoader"
            );

        if (!loader) return;


        loader.classList.remove(
            "active"
        );

    };


    /* ---------------------------------------------------------
       BACKEND CONNECTION INDICATOR
       --------------------------------------------------------- */

    window.setBackendStatus = function (
        connected
    ) {

        const indicator =
            document.querySelector(
                "#backendStatus"
            );

        if (!indicator) return;


        indicator.classList.remove(
            "backend-online",
            "backend-offline"
        );


        if (connected) {

            indicator.classList.add(
                "backend-online"
            );

            indicator.innerHTML = `

                <span
                    class="connection-light">
                </span>

                MODEL ONLINE

            `;

        }

        else {

            indicator.classList.add(
                "backend-offline"
            );

            indicator.innerHTML = `

                <span
                    class="connection-light">
                </span>

                MODEL OFFLINE

            `;

        }

    };


    /* ---------------------------------------------------------
       RESULT REVEAL

       Call this after receiving backend result.
       --------------------------------------------------------- */

    window.revealAnalysisResult = function (
        damage
    ) {

        const result =
            document.querySelector(
                "#analysisResult"
            );


        if (result) {

            result.classList.remove(
                "result-reveal"
            );

            void result.offsetWidth;

            result.classList.add(
                "result-reveal"
            );

        }


        const damageValue =
            document.querySelector(
                "#damageValue"
            );


        if (damageValue) {

            animateNumber(
                damageValue,
                damage,
                3,
                1000
            );

        }


        updateHealthStatus(damage);

        animateDamageGauge(damage);

    };


    /* ---------------------------------------------------------
       CTRL / CMD + U → FILE UPLOAD
       --------------------------------------------------------- */

    document.addEventListener(
        "keydown",
        event => {

            if (
                (event.ctrlKey || event.metaKey) &&
                event.key.toLowerCase() === "u"
            ) {

                event.preventDefault();


                const fileInput =
                    document.querySelector(
                        'input[type="file"]'
                    );


                if (fileInput) {

                    fileInput.click();

                }

            }

        }
    );


    console.log(
        "%cSHM CONTROL SYSTEM",
        "font-weight:bold; font-size:16px; color:#174b63"
    );

    console.log(
        "Cache Me If You Can / PS3"
    );

});