-- PDF (decks exported as PDF, papers, briefs) and LaTeX (Beamer decks, articles)
-- uploads. The stored kind names the uploaded format; the extracted structure
-- (slides or sections) is recorded with the document version.
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_source_kind_check;
ALTER TABLE documents ADD CONSTRAINT documents_source_kind_check
    CHECK (source_kind IN ('pptx', 'summary', 'pdf', 'latex'));
