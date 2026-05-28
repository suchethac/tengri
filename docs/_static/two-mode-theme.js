// Two-mode theme toggle: light <-> dark, never auto.
//
// Furo ships a three-state toggle (light -> dark -> auto). The auto
// state defers to the OS preference which makes the rendered theme
// inconsistent across users. This script forces the toggle to skip
// auto and only flip between explicit light and dark, and resolves
// any pre-existing "auto" state to a concrete value on load.
(function () {
    function resolveTheme() {
        const stored = localStorage.getItem("theme");
        if (stored === "light" || stored === "dark") return stored;
        return window.matchMedia("(prefers-color-scheme: dark)").matches
            ? "dark"
            : "light";
    }

    function setTheme(theme) {
        document.body.dataset.theme = theme;
        localStorage.setItem("theme", theme);
    }

    function init() {
        // Resolve auto / missing state to an explicit value.
        const current = document.body.dataset.theme;
        if (current === "auto" || current === undefined || current === "") {
            setTheme(resolveTheme());
        }

        // Intercept the toggle click in capture phase so Furo's own
        // cycle handler never sees it. Then do our two-state flip.
        const btns = document.querySelectorAll(".theme-toggle");
        btns.forEach((btn) => {
            btn.addEventListener(
                "click",
                function (e) {
                    e.preventDefault();
                    e.stopImmediatePropagation();
                    const next =
                        document.body.dataset.theme === "dark"
                            ? "light"
                            : "dark";
                    setTheme(next);
                },
                true
            );
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
