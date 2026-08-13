const footnotes = document.querySelectorAll(".inline-footnote");

function setOpen(footnote, open) {
  const trigger = footnote.querySelector(".footnote-trigger");
  footnote.classList.toggle("is-open", open);
  trigger.setAttribute("aria-expanded", String(open));
}

for (const footnote of footnotes) {
  const trigger = footnote.querySelector(".footnote-trigger");

  trigger.addEventListener("click", (event) => {
    event.stopPropagation();
    const shouldOpen = !footnote.classList.contains("is-open");
    for (const item of footnotes) setOpen(item, false);
    setOpen(footnote, shouldOpen);
  });
}

document.addEventListener("click", () => {
  for (const footnote of footnotes) setOpen(footnote, false);
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;

  for (const footnote of footnotes) {
    if (!footnote.classList.contains("is-open")) continue;
    setOpen(footnote, false);
    footnote.querySelector(".footnote-trigger").focus();
  }
});
