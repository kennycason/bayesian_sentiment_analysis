package com.kennycason.ml.classifier.bayes

import com.kennycason.nlp.gram.Gram
import org.slf4j.LoggerFactory
import java.util.*

/**
 * Created by kenny on 6/13/16.
 *
 * A Bayesian classifier.
 */
class NaiveBayesianClassifier(override val exclusions: MutableSet<String> = mutableSetOf(),
                              override val interestingGramsCount: Int = 15,
                              override val assumePrioriWhenSubjectAbsent: Boolean = false,
                              override val negativeProbabilityPriori: Float = 0.4f,
                              val model: MutableMap<String, Subject> = mutableMapOf()) : BayesianClassifier {
    private enum class ModelClass { POSITIVE, NEGATIVE }

    // apply Bayes Rule
    override fun classify(grams: List<Gram<String>>): Float {
        var positiveProbabilityProduct = 1.0f
        var negativeProbabilityProduct = 1.0f
        // for each subject, multiply p(neg) as well as 1 - p(neg)
        val interestingSubjects: List<Subject> = getInterestingSubjects(grams)
        interestingSubjects.forEach { subject ->
            positiveProbabilityProduct *= subject.negativeProbability
            negativeProbabilityProduct *= (1.0f - subject.negativeProbability)
        }
        // apply formula to calculate p(pos)
        return negativeProbabilityProduct / (positiveProbabilityProduct + negativeProbabilityProduct)
    }

    fun trainPositive(grams: List<Gram<String>>) = train(grams, ModelClass.POSITIVE)

    fun trainNegative(grams: List<Gram<String>>) = train(grams, ModelClass.NEGATIVE)

    fun finalizeTraining() {
        val totalPositiveTerms = model.values.sumBy { subject -> subject.positiveCount }
        val totalNegativeTerms = model.values.sumBy { subject -> subject.negativeCount }
        model.values.forEach { value ->
            value.finalizeProbabilities(totalPositiveTerms, totalNegativeTerms)
        }
    }

    private fun getInterestingSubjects(grams: List<Gram<String>>): List<Subject> {
        val interestingSubjects = mutableListOf<Subject>()

        grams.forEach { gram ->
            val subject = getOrCreateSubject(gram)
            if (!model.contains(subject.token) && !assumePrioriWhenSubjectAbsent) {
                return@forEach
            }
            if (interestingSubjects.isEmpty()) { // If this list is empty, then add this word in!
                interestingSubjects.add(subject)
            } else { // Otherwise, add it in sorted order by interesting level
                for (j in 0..interestingSubjects.size - 1) {
                    // for every word in the list already
                    val interestingSubject = interestingSubjects[j]
                    // if it's the same word, do nothing
                    if (subject.token == interestingSubject.token) {
                        break
                        // if it's more interesting stick it in the list
                    } else if (subject.interesting() > interestingSubject.interesting()) {
                        interestingSubjects.add(j, subject)
                        break
                        // if we get to the end, simply append
                    } else if (j == interestingSubjects.size - 1) {
                        interestingSubjects.add(subject)
                    }
                }
            }
            // If the list is bigger than the limit, delete entries at the end.
            // the more "interesting" subjects are at the start of the list
            while (interestingSubjects.size > interestingGramsCount) {
                interestingSubjects.removeAt(interestingSubjects.size - 1)
            }
        }
        return interestingSubjects
    }

    private fun getOrCreateSubject(gram: Gram<String>): Subject {
        val token = gram.buildToken()
        if (model.containsKey(token)) {
            return model[token]!!
        }
        return Subject(token = token, negativeProbability = negativeProbabilityPriori)
    }

    private fun train(grams: List<Gram<String>>, modelClass: ModelClass) {
        grams.forEach { gram ->
            val token = gram.buildToken()
            if (!model.contains(token)) {
                model[token] = Subject(token)
            }
            when (modelClass) {
                ModelClass.POSITIVE -> {
                    model[token]!!.positiveCount++
                }
                ModelClass.NEGATIVE -> {
                    model[token]!!.negativeCount++
                }
            }
        }
    }

    // default constructor values are added to facilitate auto-loading from json files via jackson in ModelPersister
    class Subject(var token: String = "",
                  var negativeCount: Int = 0, // total times it appears in "negative" messages
                  var positiveCount: Int = 0, // total times it appears in "positive" messages
                  var negativeRatio: Float = 0.0f, // negative count / total negative words
                  var positiveRatio: Float = 0.0f, // positive count / total positive words
                  var positiveProbability: Float = 0.0f, // probability this word is negative
                  var negativeProbability: Float = 0.0f) { // probability this word is positive

        fun interesting() = Math.abs(0.5f - negativeProbability)

        // implement Bayes rule to compute probability of being positive/negative
        fun finalizeProbabilities(positiveTotalCount: Int, negativeTotalCount: Int) {
            if (negativeTotalCount > 0) {
                negativeRatio = negativeCount.toFloat() / negativeTotalCount.toFloat()
            }
            if (positiveTotalCount > 0) {
                positiveRatio = positiveCount.toFloat() / positiveTotalCount.toFloat()
            }
            if (positiveRatio + negativeRatio > 0) {
                negativeProbability = negativeRatio / (negativeRatio + positiveRatio)
                positiveProbability = positiveRatio / (negativeRatio + positiveRatio)
            }
            negativeProbability = Math.min(0.99f, Math.max(0.01f, negativeProbability))
            positiveProbability = Math.min(0.99f, Math.max(0.01f, positiveProbability))
        }
    }

}