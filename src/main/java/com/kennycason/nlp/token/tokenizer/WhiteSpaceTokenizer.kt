package com.kennycason.nlp.token.tokenizer

import org.apache.commons.lang3.StringUtils
import org.eclipse.collections.api.list.ListIterable
import org.eclipse.collections.impl.factory.Lists
import org.eclipse.collections.impl.list.mutable.ListAdapter

/**
 * Created by kenny on 7/29/16.
 *
 * A very naive tokenizer
 */
class WhiteSpaceTokenizer : Tokenizer<String> {
    private val whiteSpaceRegex = Regex("[ |\n|\r]+")

    override fun tokenize(text: String): ListIterable<String> {
        if (StringUtils.isBlank(text)) { return Lists.immutable.empty() }
        return ListAdapter.adapt(text.split(whiteSpaceRegex))
    }

}