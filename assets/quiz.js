/* Reusable retrieval-practice widget, shared by every lesson.
 *
 * Markup contract:
 *   <div class="quiz" data-quiz>
 *     <p class="stem">…question…</p>
 *     <div class="opts">
 *       <button data-correct data-fb="why this is right">OPTION-A</button>
 *       <button data-fb="why this is wrong">OPTION-B</button>
 *     </div>
 *     <p class="fb" hidden></p>
 *   </div>
 *
 * Feedback is immediate (the tightest loop we can get in a static file) and the
 * correct answer is always revealed, so a wrong click still teaches. Answers are
 * deliberately equal-length so button width leaks no clue.
 *
 * An optional <div class="scorebar" data-score></div> anywhere on the page keeps
 * a running tally.
 */
(function () {
  "use strict";

  var answered = 0, correct = 0, total = 0;

  function renderScore() {
    document.querySelectorAll("[data-score]").forEach(function (el) {
      el.innerHTML = answered === 0
        ? "Retrieval practice — <b>" + total + "</b> items. Answer from memory; guessing is fine, it still builds the trace."
        : "Answered <b>" + answered + "/" + total + "</b> &nbsp;·&nbsp; correct <b>" + correct + "</b>";
    });
  }

  function wire(quiz) {
    total++;
    var buttons = Array.prototype.slice.call(quiz.querySelectorAll(".opts button"));
    var fb = quiz.querySelector(".fb");

    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (quiz.dataset.done) return;
        quiz.dataset.done = "1";
        answered++;

        var isRight = btn.hasAttribute("data-correct");
        if (isRight) correct++;

        buttons.forEach(function (b) {
          b.disabled = true;
          if (b.hasAttribute("data-correct")) b.classList.add("right");
        });
        if (!isRight) btn.classList.add("wrong");

        if (fb) {
          var own = btn.getAttribute("data-fb") || "";
          var key = isRight ? "" : (quiz.getAttribute("data-explain") || "");
          fb.innerHTML = (isRight ? "<strong>Right.</strong> " : "<strong>Not quite.</strong> ")
            + own + (key ? " " + key : "");
          fb.hidden = false;
        }
        renderScore();
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-quiz]").forEach(wire);
    renderScore();
  });
})();
