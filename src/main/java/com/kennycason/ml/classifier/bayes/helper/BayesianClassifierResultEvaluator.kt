package com.kennycason.ml.classifier.bayes.helper

import com.kennycason.ml.classifier.bayes.BayesianClassifier
import com.kennycason.nlp.gram.Gram
import org.slf4j.LoggerFactory

/**
 * Created by kenny on 6/23/16.
 */
class BayesianClassifierResultEvaluator {
    private val logger = LoggerFactory.getLogger(BayesianClassifierResultEvaluator::class.java)
    var print = true

    fun evaluate(classifier: BayesianClassifier, samples: List<List<Gram<String>>>, expected: Float, errorDelta: Float): EvaluationResult {
        val result = EvaluationResult(total = samples.size)

        for (sample in samples) {
            val positiveProbability = classifier.classify(sample)
            if (Math.abs(positiveProbability - expected) <= errorDelta) {
                result.correct++
            } else if (Math.abs(positiveProbability - expected) >= 1.0 - errorDelta) {
                result.wrong++
            }
        }
        if (print) {
            logger.info("correct: ${result.correct}, wrong: ${result.wrong}, undecided: ${samples.size - result.totalRated()}")
            logger.info("${result.correct}/${samples.size} = ${result.percentRated()}% - ${result.correct / samples.size.toFloat() * 100.0f}% accuracy, ${result.percentCorrect()}% accuracy of rated data.")
        }
        return result
    }

}

class EvaluationResult(var correct: Int = 0,
                       var wrong: Int = 0,
                       val total: Int) {
    fun totalRated() = correct + wrong
    fun percentCorrect() = correct / totalRated().toFloat() * 100.0f
    fun percentRated() = totalRated() / total.toFloat() * 100.0f
}