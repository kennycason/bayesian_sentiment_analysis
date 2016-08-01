package com.kennycason.nlp.util

import org.eclipse.collections.api.list.ListIterable
import org.eclipse.collections.impl.factory.Lists
import org.eclipse.collections.impl.factory.Sets
import java.util.*

/**
 * Created by kenny on 6/17/16.
 */
class RandomSampler {
    private val random = Random()

    fun <T> sample(list: List<T>, sampleSize: Int): List<T> {
        if (sampleSize > list.size) {
            throw IllegalArgumentException("Sample size is larger than list size")
        }
        if (sampleSize == list.size) { return list }
        val placedIndices = Sets.mutable.empty<Int>()
        val sample = Lists.mutable.empty<T>()
        while (sample.size < sampleSize) {
            val i = random.nextInt(list.size)
            if (placedIndices.contains(i)) { continue }
            sample.add(list.get(i))
        }
        return sample
    }
}