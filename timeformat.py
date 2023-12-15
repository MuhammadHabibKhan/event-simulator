def convertTime(min):

    # Calculate hours, minutes, and seconds
    hours, remainder = divmod(min, 60)
    minutes, seconds = divmod(remainder, 60)

    # Format the result as "hh:mm:ss"
    time_format = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return time_format

    # print("Time in hh:mm:ss format:", time_format)
