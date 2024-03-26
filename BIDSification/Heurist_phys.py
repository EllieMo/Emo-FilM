import fnmatch


def heur(physinfo, take=""):
    """
    Set of if .. if statements to fill BIDS names.

    It requires the user (you!) to adjust it accordingly!

    It needs an ``if`` or ``if`` statement for each file that
    needs to be processed.

    The statement will test if the ``physinfo``:
        - is similar to a string (first case), or
        - exactly matches a string (second case).
    Parameters
    ----------
    physinfo: str
        Name of an input file that should be bidsified (See Notes)
    Returns
    -------
    info: dictionary of str
        Dictionary containing BIDS keys
    Notes
    -----
    The `if ..` structure should always be similar to
    ```
    if physinfo == 'somepattern':
        info['var'] = 'somethingelse'
    ```
    or, in case it's a partial match
    ```
    if fnmatch.fnmatchcase(physinfo, '*somepattern?'):
        info['var'] = 'somethingelse'
    ```
    Where:
        - `physinfo` and `info` are dedicated keywords,
        - 'somepattern' is the name of the file,
        - 'var' is a bids key in the list below
        - 'somethingelse' is the value of the key
    """
    info = {}
    # ################################# #
    # ##        Modify here!         ## #
    # ##                             ## #
    # ##  Possible variables are:    ## #
    # ##    -info['task'] (required) ## #
    # ##    -info['run']             ## #
    # ##    -info['rec']             ## #
    # ##    -info['acq']             ## #
    # ##    -info['dir']             ## #
    # ##                             ## #
    # ##  Remember that they are     ## #
    # ##  dictionary keys            ## #
    # ##  See example below          ## #
    # ################################# #

    if physinfo == "S17_2":
        if take == "01":
            info["task"] = "LessonLearned"
        if take == "02":
            info["task"] = "AfterTheRain"
        if take == "03":
            info["task"] = "TheSecretNumber"
        if take == "04":
            info["task"] = "TearsOfSteel"

    if physinfo == "S21_2":
        if take == "01":
            info["task"] = "AfterTheRain"
        if take == "02":
            info["task"] = "ToClaireFromSonny"
        if take == "03":
            info["task"] = "TearsOfSteel"
        if take == "04":
            info["task"] = "Chatter"
        if take == "05":
            info["task"] = "LessonLearned"
    if physinfo == "S21_3":
        if take == "01":
            info["task"] = "TheSecretNumber"
        if take == "02":
            info["task"] = "Spaceman"
        if take == "03":
            info["task"] = "BetweenViewings"

    if physinfo == "S22_2":
        if take == "01":
            info["task"] = "Chatter"
        if take == "02":
            info["task"] = "LessonLearned"
        if take == "03":
            info["task"] = "BetweenViewings"
    if physinfo == "S22_3":
        if take == "01":
            info["task"] = "YouAgain"
        if take == "02":
            info["task"] = "Sintel"
        if take == "03":
            info["task"] = "FirstBite"
        if take == "04":
            info["task"] = "TearsOfSteel"

    if physinfo == "S31_1":
        if take == "01":
            info["task"] = "Rest"
        if take == "02":
            info["task"] = "ToClaireFromSonny"
        if take == "03":
            info["task"] = "LessonLearned"
        if take == "04":
            info["task"] = "FirstBite"
        if take == "05":
            info["task"] = "BigBuckBunny"
        if take == "06":
            info["task"] = "Error_1"
        if take == "07":
            info["task"] = "Error_2"
    if physinfo == "S31_2":
        if take == "01":
            info["task"] = "Spaceman"
        if take == "02":
            info["task"] = "Payload"
        if take == "03":
            info["task"] = "Chatter"
    if physinfo == "S31_3":
        if take == "01":
            info["task"] = "Superhero"
        if take == "02":
            info["task"] = "TearsOfSteel"
        if take == "03":
            info["task"] = "BetweenViewings"
        if take == "04":
            info["task"] = "AfterTheRain"
    if physinfo == "S31_4":
        if take == "01":
            info["task"] = "TheSecretNumber"
        if take == "02":
            info["task"] = "YouAgain"
        if take == "03":
            info["task"] = "Sintel"
    if physinfo == "S32_1":
        if take == "01":
            info["task"] = "Rest"
        if take == "02":
            info["task"] = "YouAgain"
        if take == "03":
            info["task"] = "BetweenViewings"
        if take == "04":
            info["task"] = "ToClaireFromSonny"
    if physinfo == "S32_2":
        if take == "01":
            info["task"] = "AfterTheRain"
        if take == "02":
            info["task"] = "FirstBite"
        if take == "03":
            info["task"] = "TearsOfSteel"
        if take == "04":
            info["task"] = "Spaceman"
    if physinfo == "S32_3":
        if take == "01":
            info["task"] = "Error"
        if take == "02":
            info["task"] = "Payload"
        if take == "03":
            info["task"] = "LessonLearned"
        if take == "04":
            info["task"] = "Chatter"
        if take == "05":
            info["task"] = "Sintel"
    if physinfo == "S32_4":
        if take == "01":
            info["task"] = "Superhero"
        if take == "02":
            info["task"] = "BigBuckBunny"
        if take == "03":
            info["task"] = "TheSecretNumber"

    return info
