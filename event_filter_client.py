def get_custom_data(models, events, distances, strokes, courses):
    """
    Prompts user for course/stroke/distance filter, then returns subset of models and events matching
    that filter.
    
    Args:
        models: a dictionary of event to model info dicts
        events: all available events
        distances: valid swim distances
        strokes: valid swim strokes
        courses: valid swim courses (SCY, SCM, LCM)

    Returns:
        a tuple (filtered_models, filtered_events, title) for the user's selections
    """
    print("Which course of events do you want information on (SCY, SCM, LCM)? Type N/A to skip. ", end="")
    course = input()
    while course not in courses and course != "n/a":
        print("Invalid course. Enter valid course or type n/a to skip. ", end="")
        course = input()

    print("Which stroke of events (FR, BK, BR, FL, IM)? Type n/a to skip. ", end="")
    stroke = input()
    while stroke not in strokes and stroke != "n/a":
        print("Invalid stroke. Enter valid stroke or type n/a to skip. ", end="")
        stroke = input()

    print("Which distance of events (50, 100, 200, 400, 500, 800, 1000, 1500, 1650) Type N/A to skip. ", end="")
    distance = input()
    while distance != "n/a" and int(distance) not in distances :
        print("Invalid distance (or not enough data on this distance to display)", end="")
        distance = input()

    filtered_models = {}
    filtered_events = []

    for event in events:

        # add all events of a specific course/stroke/distance, or all if N/A
        if ((course in event or course == "n/a")
                and (stroke in event or stroke == "n/a")
                and (distance in event or distance == "n/a")):

            filtered_events.append(event)

            if event in models:
                filtered_models[event] = models[event]

    if stroke == "n/a" and course == "n/a" and distance == "n/a":
        title = "all events"
    else:
        stroke_str = stroke if stroke != "n/a" else ""
        course_str = course if course != "n/a" else ""
        distance_str = (distance + "s" if distance != "n/a" else "")
        title = " ".join(
            part for part in [distance_str, stroke_str, course_str]
            if part
        )

    print(f"{title}: {filtered_events}")
    print("Are you alright with the selection of events? Enter 'no' to start over?")
    response = input()

    if response == 'no':
        return get_custom_data(models, events, distances, strokes, courses)

    return filtered_models, filtered_events, title

