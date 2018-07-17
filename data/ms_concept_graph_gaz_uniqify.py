#!/usr/bin/python
# -*- coding: utf8 -*-
'''
Simplify the gazetteers produced by Sebastians automatic lookup of entities with a given set of concept types such that
the gazetteer only contains the best-scoring row per entity. E.g. from rows

grep -P '^microsoft \|' sdw_bootner_organization.gaz | sort -t'|' -k7 -r:

microsoft | GTYPE:gaz_org_type | SUB:"company" | N:"6189" | C_E:"0.224085" | E_C:"0.015029" | REP:"0.003368"
microsoft | GTYPE:gaz_org_type | SUB:"company" | N:"6189" | C_E:"0.224085" | E_C:"0.015029" | REP:"0.003368"
microsoft | GTYPE:gaz_org_type | SUB:"vendor" | N:"898" | C_E:"0.032514" | E_C:"0.032180" | REP:"0.001046"
microsoft | GTYPE:gaz_org_type | SUB:"vendor" | N:"898" | C_E:"0.032514" | E_C:"0.032180" | REP:"0.001046"
microsoft | GTYPE:gaz_org_type | SUB:"software company" | N:"279" | C_E:"0.010102" | E_C:"0.093682" | REP:"0.000946"

only keep the first row.

The script assumes that the input file is sorted by column 7, e.g. using "sort -t'|' -k7 -r infile.gaz > out_sorted.gaz"

Subtypes collected with cat sdw_bootner_organization.gaz | awk -F'|' '{print $3}' | sort | uniq > org_subtypes.txt
and then filtered manually. Or ask Sebastian to provide the list of concepts he used to create the gazetteer.

@date: 03.07.18
@author: leonhard.hennig@dfki.de
'''
# from http://stackoverflow.com/questions/2276200/changing-default-encoding-of-python
import sys
import codecs
import re
from tqdm import tqdm


pattern_org_concepts = re.compile('SUB:"(academic institution|advertiser|agency|aggregator|agricultural university|'
                              'air carrier|airline|american company|analyst|android phone maker|apple s competitor|'
                              'apple supplier|appliance brand|appliance company|appliance firm|area corporation|'
                              'auto company|auto giant|automaker|automobile manufacturer|automotive safety system '
                              'supplier|automotive supplier|aviation company|aws customer|band|bank|best brand|'
                              'big bank|big brand|big company|biggest studio|big global brand|big investment bank|'
                              'big manufacturer|big name client|big name company|big player|big studio|big tech '
                              'company|biometric maker|biotech company|biotechnology company|british manufacturer|'
                              'broadcaster|brokerage|brokerage firm|broking house|building automation company|business'
                              '|camera maker|canadian company|car company|card issuer|carrier|carrying top brand|'
                              'casino|centre|ces company|chinese and japanese firm|chip maker|chipmakers|city|client|'
                              'club|college|college campus|commercial press release distribution service|company|'
                              'competitor|component manufacturer|conglomerate|connotate s customer|conservative '
                              'analyst|construction management software vendor|consumer company|contender|content '
                              'provider|corp oration|corporation|courier|credit repair company|cultural center|customer|'
                              'dealer|defense company|defense contractor|developer|device manufacturer|distribution '
                              'company|distributor|edge company|electronic giant|electronics maker|employer|energy '
                              'company|energy firm|energy project construction company|energy provider|engine company|'
                              'engineering company|engineering consultancy|engineering firm|enterprise|entertainment '
                              'giant|equipment maker|equipment supplier|equipment vendor|esco|established company|'
                              'established player|established supplier|established vendor|esteemed company|esteemed '
                              'organization|ethanol producer|european company|european utility|exhibitor|exporter|'
                              'fabless company|federal agency|financial giant|financial institution|firewall vendor|'
                              'firm|florida energy company|fracking pioneer|franchise|german company|german factory|'
                              'giant|giant engineering contractor|global bank|global brand|global company|global '
                              'corporation|global enterprise|global household name|global leader|global market leader|'
                              'global new company|global news agency|global oem|global player|good young player|'
                              'government agency|governmental agency|government authority|government entity|gpu vendor|'
                              'grantmakers|group|gsm communication equipment provider|hard disk manufacturer|high '
                              'performance manufacturer|high performing school|high power semiconductor|high profile '
                              'client|high profile company|high voltage device|hi tech company|hollywood studio|'
                              'household name|huge company|ic manufacturer|identifying partner|imported car '
                              'manufacturer|indian group|industrial company|industrial giant|industrial partner|'
                              'industry|industry giant|industry leader|industry-leading technology provider|industry '
                              'manufacturer|industry organization|industry pioneer|industry publication|industry '
                              'stalwart|industry supplier|innovator|institution|instrument vendor|insurance company|'
                              'insurer|integrated device manufacturer|intermediary|international agency|international '
                              'bank|international client|international company|international component manufacturer|'
                              'international corporation|international giant|international news outlet|international '
                              'organization|international publication|international renowned brand|international '
                              'semiconductor manufacturer|internet based company|internet company|investment bank|'
                              'investment firm|investment house|investor|investor-owned utility|japanese automotive '
                              'supplier|japanese company|key firm|key industry manufacturer|key player|key third party '
                              'client|laboratory|large bank|large business|large company|large corporate client|large '
                              'corporation|large datacenter equipment supplier|large employer|large firm|large '
                              'international player|large investor|large japanese corporation|large manufacturer|'
                              'large multinational company|large multinational corporation|large organization|'
                              'large player|large regional bank|large retailer|large technology company|large utility|'
                              'large well known company|leading analyst firm|leading company|leading corporate|leading '
                              'edge company|leading industry analyst firm|leading investment house|leading japanese '
                              'company|leading manufacturer|leading medium company|leading player|leading vendor|lender'
                              '|licensee|local company|localization front runner|log loader|long established franchise'
                              '|luxury car maker|luxury marketer|magazine|maker|manufacturer|manufacturing company|'
                              'market leader|market participant|market player|market researcher|market research '
                              'organization|mcu vendor|medical device company|medium company|medium giant|medium outlet|'
                              'mobile giant|mobile manufacturer|multinational company|multinational corporation|'
                              'multinational firm|multinational organization|multinational powerhouse|national lab|'
                              'national laboratory|national organization|nation\'s largest bank|natural gas producer|'
                              'natural gas supplier|natural gas weighted e&p company|network equipment manufacturer|'
                              'news agency|news clipping service|news organization|news outlet|newspaper|news '
                              'publication|news service|news source|newswire|news wire service|newswire service|'
                              'north american and european supplier|notable company|notable competitor|notable employer'
                              '|oem|oil company|oil firm|online news service|online press release service|operator|'
                              'organisation|organization|original equipment manufacturer|outstanding growth '
                              'entrepreneur|overseas company|pharmaceutical company|photo agency|power company|press '
                              'release distribution service|press release service|prime contractor|printer brand|'
                              'private company|private hospital|private organization|private school|producer|'
                              'progressive company|prospective competitor|provider|public company|publisher|pv module '
                              'manufacturer|quality conscious company|rail line|real estate company|reason company|'
                              'regional bank|regional firm|registry|regulator|renowned company|renowned manufacturer|'
                              'research organization|respected company|respected manufacturer|retailer|safety net '
                              'provider|school|scientific organization|semiconductor company|semiconductor supplier|'
                              'semiconductor vendor|service provider|shareholder|silicon vendor|small company|small '
                              'institute|smartphone company|social medium outlet|software company|stakeholder|start up|'
                              'startup|state agency|strategic partner|strategic player|streaming company|studio|'
                              'supermarket chain|supplier|supplier lead carrier service|tech company|tech giant|'
                              'technological company|technology company|technology driver|technology firm|technology '
                              'provider|tech player|telecom company|test equipment manufacturer|tier one supplier|'
                              'top-class organization|top company|toronto broking firm|traditional medium company|'
                              'transceiver manufacturer|transformer vendor|transportation network company|university|'
                              'u s company|u.s. company|u s rival|utility|utility company|vendor|web service provider|'
                              'well known company|well-known company|western company|west european company|wind energy '
                              'company|wire service|world class educational institution|world renowned company)"')



def print_best_row_per_entity(infile, outfile):
    known_entities = set([])
    counter = 0
    removed_by_subtype = 0
    print('Processing input [%s], writing to [%s]' % (infile, outfile))
    with codecs.open(outfile, 'wb', 'utf8') as output:
        with codecs.open(infile, 'rb', 'utf8') as input:
            for line in tqdm(input):
                entity = line.split("|")[0].strip()
                gaz_type = line.split("|")[1].strip()
                sub_type = line.split("|")[2].strip()
                counter = counter + 1
                if filter_by_subtype(gaz_type, sub_type):
                    if entity not in known_entities:
                        output.write(line)
                        known_entities.add(entity)
                else:
                    removed_by_subtype = removed_by_subtype + 1
    print('Processed [%d] total lines, wrote [%d] unique entities (lines removed by subtype: [%d]' % (counter + 1,
                                                                            len(known_entities), removed_by_subtype))

def filter_by_subtype(gaz_type, sub_type):
    """
    since bash grep couldn't handle this regex string
    :param infile:
    :param outfile:
    :return:
    """

    #if gaz_type == "GTYPE:gaz_org_type":
    #    return pattern_org_concepts.match(sub_type)
    return True

types = ['organization', 'product']
#types = ['task', 'material', 'process']
for ne_type in types:
    basedir = 'concepts'
    infile = '%s/sdw_bootner_%s_sorted.gaz' % (basedir, ne_type)
    outfile = '%s/sdw_bootner_%s.gaz' % (basedir, ne_type)
    print_best_row_per_entity(infile, outfile)
