import re
from typing import Optional
from podium.preproc import TextCleanUp


class IronyDetectionPreprocessor(TextCleanUp):
    """
    Simple decorator for the TextCleanUp class.
    """
    def __init__(self,
                 language="en",
                 fix_unicode: bool = True,
                 to_ascii: bool = True,
                 remove_line_breaks: bool = False,
                 remove_punct: bool = False,
                 replace_url: Optional[str] = None,
                 replace_email: Optional[str] = None,
                 replace_phone_number: Optional[str] = None,
                 replace_number: Optional[str] = None,
                 replace_digit: Optional[str] = None,
                 replace_currency_symbol: Optional[str] = None,
                 ellipsis=True,
                 ) -> None:
        super(IronyDetectionPreprocessor, self).__init__(
            language=language,
            fix_unicode=fix_unicode,
            to_ascii=to_ascii,
            remove_line_breaks=remove_line_breaks,
            remove_punct=remove_punct,
            replace_url=replace_url,
            replace_email=replace_email,
            replace_phone_number=replace_phone_number,
            replace_number=replace_number,
            replace_digit=replace_digit,
            replace_currency_symbol=replace_currency_symbol,
        )
        self.ellipsis = ellipsis
        self.remove_punct = remove_punct

    def __call__(self, raw: str) -> str:
        if self.ellipsis and not self.remove_punct:
            raw = re.sub(r"\.{2,}", "...", raw)
        return self._cleanup(raw)


# test
if __name__ == '__main__':
    single_ellipsis = 'please do..i need the second hand embarrassment so desperatly on my phone '
    multiple_ellipsis = 'please do..i need the second .. hand embarrassment so ..desperatly on my phone '
    different_kinds_ellipsis = 'please do..i....... need the second ........ hand embarrassment so ...desperatly on ' \
                               'my phone '

    preprocessor = IronyDetectionPreprocessor()

    result = preprocessor(single_ellipsis)
    assert (result == 'please do...i need the second hand embarrassment so desperatly on my phone')

    result = preprocessor(multiple_ellipsis)
    assert (result == 'please do...i need the second ... hand embarrassment so ...desperatly on my phone')

    result = preprocessor(different_kinds_ellipsis)
    assert (result == 'please do...i... need the second ... hand embarrassment so ...desperatly on my phone')
