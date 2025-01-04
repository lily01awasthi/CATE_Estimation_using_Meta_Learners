# Benchmark Dataset Description

## Benchmark Dataset for Validation

The benchmark dataset, named **CausalEdu**, is derived from the Eedi online learning platform and designed specifically for causal discovery and inference tasks. This real-world education dataset captures student interactions with multiple-choice questions, focusing on mathematical constructs. It includes both observational data and ground truth values, enabling robust training and evaluation of meta-learners for estimating Conditional Average Treatment Effects (CATE).

### Dataset Composition

The **CausalEdu** dataset comprises two key components integrated for this study:

#### Training Data
The training data includes detailed records of student activities during diagnostic quizzes, encompassing "Checkin" and "Checkout" questions, as well as lessons in between. Each record includes the correctness of responses and contextual information such as the concept tested, question sequence, and activity type.

Key covariates in the Training Data include:

- **`QuizSessionId`**: Identifies a student's quiz attempt.
- **`AnswerId`**: ID for each individual answer.
- **`UserId`**: Unique student identifier.
- **`QuizId`**: Identifies each quiz.
- **`QuestionId`**: Identifies each question.
- **`IsCorrect`**: Binary (1 = correct, 0 = incorrect).
- **`AnswerValue`**: The student's selected answer (1-4).
- **`CorrectAnswer`**: The correct answer (1-4).
- **`QuestionSequence`**: The sequence of questions in the quiz (1-5).
- **`ConstructId`**: The concept being tested by the question.
- **`Type`**: Type of activity (Checkin, Checkout, Lesson, etc).
- **`Timestamp`**: Time of the quiz activity.

#### Ground Truth Data
Derived from A/B tests conducted on the Eedi platform, this component provides experimentally validated CATE values for two specific hypotheses regarding the causal effects of mastering one construct on performance in related constructs.

Key covariates in the Ground Truth Data include:

- **`TreatmentLessonConstructId`**: The construct taught to the treatment group.
- **`QuestionConstructId`**: The target construct being measured for treatment effects.
- **`Year`**: The year group of students involved in the experiment.
- **`ControlLessonConstructId`**: The construct taught to the control group.
- **`ControlUsersCount`**: Number of students in the control group.
- **`TreatmentUsersCount`**: Number of students in the treatment group.
- **`ate_p_1__`**: Ground truth CATE for Hypothesis 1 (Excluding students who answered the check-in correctly but the check-out incorrectly ).
- **`ate_k_1__`**: Ground truth CATE for Hypothesis 2 (Including students who answered the check-in
correctly but the check-out incorrectly as ”random guessing).

### Data Integration and Propensity Score Matching (PSM)

The training and ground truth data are merged based on `ConstructId` and `QuestionConstructId`, aligning student quiz interactions with corresponding experimental conditions. Only relevant events such as "Checkin", "Checkout", "IsCorrect" are included.


