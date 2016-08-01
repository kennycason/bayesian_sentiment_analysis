package com.kennycason.ml.classifier.bayes

import com.kennycason.nlp.gram.Gram

/**
 * Created by kenny on 6/17/16.
 */
interface BayesianClassifier {
    val exclusions: MutableSet<String>
    val interestingGramsCount: Int
    val assumePrioriWhenSubjectAbsent: Boolean
    val negativeProbabilityPriori: Float

    fun classify(grams: List<Gram<String>>): Float
}