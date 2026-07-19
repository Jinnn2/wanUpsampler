-- Keep Pandoc's mechanical Markdown-to-LaTeX conversion compatible with the
-- AAAI template, which forbids hyperlink packages. Cross-references are added
-- explicitly in the curated LaTeX source.
function Header(el)
  el.identifier = ""
  el.attributes = {}
  return el
end

function Div(el)
  el.identifier = ""
  el.attributes = {}
  return el
end

function Span(el)
  el.identifier = ""
  el.attributes = {}
  return el
end
