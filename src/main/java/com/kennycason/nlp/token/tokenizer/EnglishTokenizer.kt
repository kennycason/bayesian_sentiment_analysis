package com.kennycason.nlp.token.tokenizer

import org.apache.commons.lang3.StringUtils
import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import org.apache.lucene.util.Version
import org.eclipse.collections.api.list.ListIterable
import org.eclipse.collections.impl.factory.Lists
import org.eclipse.collections.impl.list.mutable.ListAdapter
import java.io.IOException
import java.io.StringReader

/**
 * Created by kenny on 7/29/16.
 *
 * A better tokenizer utilizing Lucene's language analyzers.
 */
class EnglishTokenizer : Tokenizer<String> {
    private val standardAnalyzer = StandardAnalyzer(Version.LUCENE_36)

    override fun tokenize(text: String): ListIterable<String> {
        if (StringUtils.isBlank(text)) { return Lists.immutable.empty() }
        val tokens = Lists.mutable.empty<String>()
        try {
            val stream = standardAnalyzer.tokenStream(null, StringReader(text))
            stream.reset()
            while (stream.incrementToken()) {
                tokens.add(stream.getAttribute(CharTermAttribute::class.java).toString())
            }
        } catch (e: IOException) {
            throw RuntimeException(e)
        }

        return tokens
    }


}