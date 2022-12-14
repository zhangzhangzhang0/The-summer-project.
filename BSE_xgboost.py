import pickle
import sys
import math
import random

import pandas as pd

bse_sys_minprice = 1  # minimum price in the system, in cents/pennies
bse_sys_maxprice = 1000  # maximum price in the system, in cents/pennies
ticksize = 1  # minimum change in price, in cents/pennies


# an Order/quote has a trader id, a type (buy/sell) price, quantity, timestamp, and unique i.d.
class Order:

    def __init__(self, tid, otype, price, qty, time, qid):
        self.tid = tid  # trader i.d.
        self.otype = otype  # order type
        self.price = price  # price
        self.qty = qty  # quantity
        self.time = time  # timestamp
        self.qid = qid  # quote i.d. (unique to each quote)

    def __str__(self):
        return '[%s %s P=%03d Q=%s T=%5.2f QID:%d]' % \
               (self.tid, self.otype, self.price, self.qty, self.time, self.qid)


# Orderbook_half is one side of the book: a list of bids or a list of asks, each sorted best-first

class Orderbook_half:

    def __init__(self, booktype, worstprice):
        # booktype: bids or asks?
        self.booktype = booktype
        # dictionary of orders received, indexed by Trader ID
        self.orders = {}
        # limit order book, dictionary indexed by price, with order info
        self.lob = {}
        # anonymized LOB, lists, with only price/qty info
        self.lob_anon = []
        # summary stats
        self.best_price = None
        self.best_tid = None
        self.worstprice = worstprice
        self.session_extreme = None  # most extreme price quoted in this session
        self.n_orders = 0  # how many orders?
        self.lob_depth = 0  # how many different prices on lob?

    def anonymize_lob(self):
        # anonymize a lob, strip out order details, format as a sorted list
        # NB for asks, the sorting should be reversed
        self.lob_anon = []
        for price in sorted(self.lob):
            qty = self.lob[price][0]
            self.lob_anon.append([price, qty])

    def build_lob(self):
        lob_verbose = False
        # take a list of orders and build a limit-order-book (lob) from it
        # NB the exchange needs to know arrival times and trader-id associated with each order
        # returns lob as a dictionary (i.e., unsorted)
        # also builds anonymized version (just price/quantity, sorted, as a list) for publishing to traders
        # a = open('lob.txt', 'w')
        self.lob = {}
        for tid in self.orders:
            order = self.orders.get(tid)
            price = order.price
            if price in self.lob:
                # update existing entry
                qty = self.lob[price][0]
                orderlist = self.lob[price][1]
                orderlist.append([order.time, order.qty, order.tid, order.qid])
                self.lob[price] = [qty + order.qty, orderlist]
                # a.write(' %s, %s, %s, %s, %s\n'%(order.time,self.lob, 'Qty', 'Party1', 'Party2'))
            else:
                # create a new dictionary entry
                self.lob[price] = [order.qty, [[order.time, order.qty, order.tid, order.qid]]]
        # print(self.lob)
        # create anonymized version
        self.anonymize_lob()
        # record best price and associated trader-id
        if len(self.lob) > 0:
            if self.booktype == 'Bid':
                self.best_price = self.lob_anon[-1][0]
            else:
                self.best_price = self.lob_anon[0][0]
            self.best_tid = self.lob[self.best_price][1][0][2]
        else:
            self.best_price = None
            self.best_tid = None

        if lob_verbose:
            print(self.lob)

    def book_add(self, order):
        # add order to the dictionary holding the list of orders
        # either overwrites old order from this trader
        # or dynamically creates new entry in the dictionary
        # so, max of one order per trader per list
        # checks whether length or order list has changed, to distinguish addition/overwrite
        # print('book_add > %s %s' % (order, self.orders))

        # if this is an ask, does the price set a new extreme-high record?
        if (self.booktype == 'Ask') and ((self.session_extreme is None) or (order.price > self.session_extreme)):
            self.session_extreme = int(order.price)

        n_orders = self.n_orders
        self.orders[order.tid] = order
        self.n_orders = len(self.orders)
        self.build_lob()
        # print('book_add < %s %s' % (order, self.orders))
        if n_orders != self.n_orders:
            return 'Addition'
        else:
            return 'Overwrite'

    def book_del(self, order):
        # delete order from the dictionary holding the orders
        # assumes max of one order per trader per list
        # checks that the Trader ID does actually exist in the dict before deletion
        # print('book_del %s',self.orders)
        if self.orders.get(order.tid) is not None:
            del (self.orders[order.tid])
            self.n_orders = len(self.orders)
            self.build_lob()
        # print('book_del %s', self.orders)

    def delete_best(self):
        # delete order: when the best bid/ask has been hit, delete it from the book
        # the TraderID of the deleted order is return-value, as counterparty to the trade
        best_price_orders = self.lob[self.best_price]
        best_price_qty = best_price_orders[0]
        best_price_counterparty = best_price_orders[1][0][2]
        if best_price_qty == 1:
            # here the order deletes the best price
            del (self.lob[self.best_price])
            del (self.orders[best_price_counterparty])
            self.n_orders = self.n_orders - 1
            if self.n_orders > 0:
                if self.booktype == 'Bid':
                    self.best_price = max(self.lob.keys())
                else:
                    self.best_price = min(self.lob.keys())
                self.lob_depth = len(self.lob.keys())
            else:
                self.best_price = self.worstprice
                self.lob_depth = 0
        else:
            # best_bid_qty>1 so the order decrements the quantity of the best bid
            # update the lob with the decremented order data
            self.lob[self.best_price] = [best_price_qty - 1, best_price_orders[1][1:]]

            # update the bid list: counterparty's bid has been deleted
            del (self.orders[best_price_counterparty])
            self.n_orders = self.n_orders - 1
        self.build_lob()
        return best_price_counterparty


# Orderbook for a single instrument: list of bids and list of asks

class Orderbook(Orderbook_half):

    def __init__(self):
        self.bids = Orderbook_half('Bid', bse_sys_minprice)
        self.asks = Orderbook_half('Ask', bse_sys_maxprice)
        self.tape = []
        self.tape_length = 1000  # maximum number of items on the tape
        self.quote_id = 0  # unique ID code for each quote accepted onto the book

    def midprice(self, bid_p, bid_q, ask_p, ask_q):
        # returns midprice as mean of best bid and best ask if both best bid & best ask exist
        # if only one best price exists, returns that as mid
        # if neither best price exists, returns None
        mprice = None
        if bid_q > 0 and ask_q == None:
            mprice = bid_p
        elif ask_q > 0 and bid_q == None:
            mprice = ask_p
        elif bid_q > 0 and ask_q > 0:
            mprice = (bid_p + ask_p) / 2.0
        return mprice

    def microprice(self, bid_p, bid_q, ask_p, ask_q):
        mprice = None
        if bid_q > 0 and ask_q > 0:
            tot_q = bid_q + ask_q
            mprice = ((bid_p * ask_q) + (ask_p * bid_q)) / tot_q
        return mprice


# Exchange's internal orderbook

class Exchange(Orderbook):

    def add_order(self, order, verbose):
        # add a quote/order to the exchange and update all internal records; return unique i.d.
        order.qid = self.quote_id
        self.quote_id = order.qid + 1
        # if verbose : print('QUID: order.quid=%d self.quote.id=%d' % (order.qid, self.quote_id))
        if order.otype == 'Bid':
            response = self.bids.book_add(order)
            best_price = self.bids.lob_anon[-1][0]
            self.bids.best_price = best_price
            self.bids.best_tid = self.bids.lob[best_price][1][0][2]
        else:
            response = self.asks.book_add(order)
            best_price = self.asks.lob_anon[0][0]
            self.asks.best_price = best_price
            self.asks.best_tid = self.asks.lob[best_price][1][0][2]
        return [order.qid, response]

    def del_order(self, time, order, verbose):
        # delete a trader's quot/order from the exchange, update all internal records
        if order.otype == 'Bid':
            self.bids.book_del(order)
            if self.bids.n_orders > 0:
                best_price = self.bids.lob_anon[-1][0]
                self.bids.best_price = best_price
                self.bids.best_tid = self.bids.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.bids.best_price = None
                self.bids.best_tid = None
            cancel_record = {'type': 'Cancel', 'time': time, 'order': order}
            self.tape.append(cancel_record)
            self.tape = self.tape[-self.tape_length:]  # right-truncate the tape

        elif order.otype == 'Ask':
            self.asks.book_del(order)
            if self.asks.n_orders > 0:
                best_price = self.asks.lob_anon[0][0]
                self.asks.best_price = best_price
                self.asks.best_tid = self.asks.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.asks.best_price = None
                self.asks.best_tid = None
            cancel_record = {'type': 'Cancel', 'time': time, 'order': order}
            self.tape.append(cancel_record)
            self.tape = self.tape[-self.tape_length:]  # right-truncate the tape
        else:
            # neither bid nor ask?
            sys.exit('bad order type in del_quote()')

    def process_order2(self, time, order, verbose):
        # receive an order and either add it to the relevant LOB (ie treat as limit order)
        # or if it crosses the best counterparty offer, execute it (treat as a market order)
        oprice = order.price
        counterparty = None

        [qid, response] = self.add_order(order, verbose)  # add it to the order lists -- overwriting any previous order
        order.qid = qid

        if verbose:
            print('QUID: order.quid=%d' % order.qid)
            print('RESPONSE: %s' % response)
        best_ask = self.asks.best_price
        best_ask_tid = self.asks.best_tid
        best_bid = self.bids.best_price
        best_bid_tid = self.bids.best_tid
        '''
        print(self.bids.best_price)
        print(self.bids.lob_anon[0])
        best_bid_q = self.bids.lob_anon[0][1]
        best_ask_q = self.asks.lob_anon[0][1]
        # 1
        output_time = time
        # 3
        output_price = order.price
        # 4
        output_spread = self.asks.best_price - self.bids.best_price
        # 5
        output_midprice = self.midprice(self.bids.best_price, best_bid_q, self.asks.best_price, best_ask_q)
        # 6
        output_microprice = self.microprice(self.bids.best_price, best_bid_q, self.asks.best_price, best_ask_q)
        # 7
        output_best_bids = self.bids.best_price
        # 8
        output_best_asks = self.asks.best_price

'''
        if order.otype == 'Bid':
            if self.asks.n_orders > 0 and best_bid >= best_ask:
                # bid lifts the best ask
                if verbose:
                    print("Bid $%s lifts best ask" % oprice)
                counterparty = best_ask_tid
                price = best_ask  # bid crossed ask, so use ask price
                if verbose:
                    print('counterparty, price', counterparty, price)

                # delete the ask just crossed
                self.asks.delete_best()
                # delete the bid that was the latest order
                self.bids.delete_best()
        elif order.otype == 'Ask':
            if self.bids.n_orders > 0 and best_ask <= best_bid:
                # ask hits the best bid
                if verbose:
                    print("Ask $%s hits best bid" % oprice)
                # remove the best bid
                counterparty = best_bid_tid
                price = best_bid  # ask crossed bid, so use bid price
                if verbose:
                    print('counterparty, price', counterparty, price)

                # delete the bid just crossed, from the exchange's records
                self.bids.delete_best()
                # delete the ask that was the latest order, from the exchange's records
                self.asks.delete_best()

        else:
            # we should never get here
            sys.exit('process_order() given neither Bid nor Ask')
        # NB at this point we have deleted the order from the exchange's records
        # but the two traders concerned still have to be notified
        if verbose:
            print('counterparty %s' % counterparty)
        if counterparty is not None:
            # process the trade
            if verbose: print('>>>>>>>>>>>>>>>>>TRADE t=%010.3f $%d %s %s' % (time, price, counterparty, order.tid))
            transaction_record = {'type': 'Trade',
                                  'time': time,
                                  'price': price,
                                  'party1': counterparty,
                                  'party2': order.tid,
                                  'qty': order.qty
                                  }
            self.tape.append(transaction_record)
            self.tape = self.tape[-self.tape_length:]  # right-truncate the tape
            return transaction_record
        else:
            return None

    # Currently tape_dump only writes a list of transactions (ignores cancellations)
    def tape_dump(self, fname, dumpfile, fmode, tmode, traders):
        # dumpfile = open(fname, fmode)
        dump2 = open(fname, fmode)
        dump2.write(' %s, %s, %s, %s, %s\n' % ('Time', 'Price', 'Qty', 'Party1', 'Party2'))
        for tapeitem in self.tape:
            if tapeitem['type'] == 'Trade':
                # dumpfile.write(' %s, %s, %s, %s, %s, %s\n' % (tapeitem['time'], tapeitem['price'], tapeitem['qty'],traders[tapeitem['party2']].ttype, traders[tapeitem['party1']].ttype, str(tapeitem) ))
                dump2.write(' %s, %s, %s, %s, %s\n' % (
                tapeitem['time'], tapeitem['price'], tapeitem['qty'], tapeitem['party2'], tapeitem['party1']))
                dumpfile.write(' %s, %s, %s, %s, %s\n' % (
                tapeitem['time'], tapeitem['price'], tapeitem['qty'], tapeitem['party2'], tapeitem['party1']))
        # dumpfile.close()
        dump2.close()
        if tmode == 'wipe':
            self.tape = []

    # this returns the LOB data "published" by the exchange,
    # i.e., what is accessible to the traders
    def publish_lob(self, time, verbose):
        public_data = {}

        public_data['time'] = time
        public_data['bids'] = {'best': self.bids.best_price,
                               'worst': self.bids.worstprice,
                               'n': self.bids.n_orders,
                               'lob': self.bids.lob_anon}
        public_data['asks'] = {'best': self.asks.best_price,
                               'worst': self.asks.worstprice,
                               'sess_hi': self.asks.session_extreme,
                               'n': self.asks.n_orders,
                               'lob': self.asks.lob_anon}
        public_data['QID'] = self.quote_id
        public_data['tape'] = self.tape
        public_data['midprice'] = 0
        public_data['microprice'] = 0
        if self.bids.n_orders > 0:
            best_bid_p = self.bids.lob_anon[0][0]
        else:
            best_bid_p = None

        if self.asks.n_orders > 0:
            best_ask_p = self.asks.lob_anon[0][0]
        else:
            best_ask_p = None
        if self.bids.n_orders > 0 and self.asks.n_orders > 0:
            # neither side of the LOB is empty
            best_bid_q = self.bids.lob_anon[0][1]
            best_ask_q = self.asks.lob_anon[0][1]
            public_data['midprice'] = self.midprice(best_bid_p, best_bid_q, best_ask_p, best_ask_q)
            public_data['microprice'] = self.microprice(best_bid_p, best_bid_q, best_ask_p, best_ask_q)

        if verbose:
            print('publish_lob: t=%d' % time)
            print('BID_lob=%s' % public_data['bids']['lob'])
            # print('best=%s; worst=%s; n=%s ' % (self.bids.best_price, self.bids.worstprice, self.bids.n_orders))
            print('ASK_lob=%s' % public_data['asks']['lob'])
            # print('qid=%d' % self.quote_id)

        return public_data


##################--Traders below here--#############


# Trader superclass
# all Traders have a trader id, bank balance, blotter, and list of orders to execute
class Trader:

    def __init__(self, ttype, tid, balance, time):
        self.ttype = ttype  # what type / strategy this trader is
        self.tid = tid  # trader unique ID code
        self.balance = balance  # money in the bank
        self.blotter = []  # record of trades executed
        self.blotter_length = 100  # maximum number of items held on a blotter
        self.orders = []  # customer orders currently being worked (fixed at 1)
        self.n_quotes = 0  # number of quotes live on LOB
        self.birthtime = time  # used when calculating age of a trader/strategy
        self.profitpertime = 0  # profit per unit time
        self.n_trades = 0  # how many trades has this trader done?
        self.lastquote = None  # record of what its last quote was

    def __str__(self):
        return '[TID %s type %s balance %s blotter %s orders %s n_trades %s profitpertime %s]' \
               % (self.tid, self.ttype, self.balance, self.blotter, self.orders, self.n_trades, self.profitpertime)

    def add_order(self, order, verbose):
        # in this version, trader has at most one order,
        # if allow more than one, this needs to be self.orders.append(order)
        if self.n_quotes > 0:
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
        self.orders = [order]
        if verbose:
            print('add_order < response=%s' % response)
        return response

    def del_order(self, order):
        # this is lazy: assumes each trader has only one customer order with quantity=1, so deleting sole order
        # CHANGE TO DELETE THE HEAD OF THE LIST AND KEEP THE TAIL
        self.orders = []

    def bookkeep(self, trade, order, verbose, time):

        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        self.blotter = self.blotter[-self.blotter_length:]  # right-truncate to keep to length

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            sys.exit()

        if verbose: print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
        self.del_order(order)  # delete the order

    # specify how trader responds to events in the market
    # this is a null action, expect it to be overloaded by specific algos
    def respond(self, time, lob, trade, verbose):
        return None

    # specify how trader mutates its parameter values
    # this is a null action, expect it to be overloaded by specific algos
    def mutate(self, time, lob, trade, verbose):
        return None


# Trader subclass Giveaway
# even dumber than a ZI-U: just give the deal away
# (but never makes a loss)
class Trader_Giveaway(Trader):

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            order = None
        else:
            quoteprice = self.orders[0].price
            order = Order(self.tid,
                          self.orders[0].otype,
                          quoteprice,
                          self.orders[0].qty,
                          time, lob['QID'])
            self.lastquote = order
        return order


# Trader subclass ZI-C
# After Gode & Sunder 1993
class Trader_ZIC(Trader):

    def getorder(self, time, countdown, lob):
        # print(lob)
        # print()
        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            minprice = lob['bids']['worst']
            maxprice = lob['asks']['worst']
            qid = lob['QID']
            limit = self.orders[0].price
            otype = self.orders[0].otype
            if otype == 'Bid':
                quoteprice = random.randint(minprice, limit)
            else:
                quoteprice = random.randint(limit, maxprice)
                # NB should check it == 'Ask' and barf if not
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, qid)
            self.lastquote = order
        return order


# Trader subclass Shaver
# shaves a penny off the best price
# if there is no best price, creates "stub quote" at system max/min
class Trader_Shaver(Trader):

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            order = None
        else:
            limitprice = self.orders[0].price
            otype = self.orders[0].otype
            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    quoteprice = lob['bids']['best'] + 1
                    if quoteprice > limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    quoteprice = lob['asks']['best'] - 1
                    if quoteprice < limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['asks']['worst']
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order


# Trader subclass Sniper
# Based on Shaver,
# "lurks" until time remaining < threshold% of the trading session
# then gets increasing aggressive, increasing "shave thickness" as time runs out
class Trader_Sniper(Trader):

    def getorder(self, time, countdown, lob):
        lurk_threshold = 0.2
        shavegrowthrate = 3
        shave = int(1.0 / (0.01 + countdown / (shavegrowthrate * lurk_threshold)))
        if (len(self.orders) < 1) or (countdown > lurk_threshold):
            order = None
        else:
            limitprice = self.orders[0].price
            otype = self.orders[0].otype

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    quoteprice = lob['bids']['best'] + shave
                    if quoteprice > limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    quoteprice = lob['asks']['best'] - shave
                    if quoteprice < limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['asks']['worst']
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order


# Trader subclass PRZI
# added 23 March 2021
# Dave Cliff's Parameterized-Response Zero-Intelligence (PRZI) trader
# see https://arxiv.org/abs/2103.11341
class Trader_PRZI(Trader):

    def __init__(self, ttype, tid, balance, time):
        # PRZI strategy defined by parameter "strat"
        # here this is randomly assigned
        # strat * direction = -1 = > GVWY; =0 = > ZIC; =+1 = > SHVR

        Trader.__init__(self, ttype, tid, balance, time)
        self.theta0 = 100  # threshold-function limit value
        self.m = 4  # tangent-function multiplier
        self.strat = 1.0 - 2 * random.random()  # strategy parameter: must be in range [-1.0, +1.0]
        self.cdf_lut_bid = None  # look-up table for buyer cumulative distribution function
        self.cdf_lut_ask = None  # look-up table for buyer cumulative distribution function
        self.pmax = None  # this trader's estimate of the maximum price the market will bear
        self.pmax_c_i = math.sqrt(random.randint(1, 10))  # multiplier coefficient when estimating p_max

    def getorder(self, time, countdown, lob):

        # shvr_price tells us what price a SHVR would quote in these circs
        def shvr_price(otype, limit, lob):

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    shvr_p = lob['bids']['best'] + 1  # BSE tick size is always 1
                    if shvr_p > limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    shvr_p = lob['asks']['best'] - 1  # BSE tick size is always 1
                    if shvr_p < limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['asks']['worst']

            return shvr_p

        # calculate cumulative distribution function (CDF) look-up table (LUT)
        def calc_cdf_lut(strat, t0, m, dirn, pmin, pmax):
            # set parameter values and calculate CDF LUT
            # dirn is direction: -1 for buy, +1 for sell

            # the threshold function used to clip
            def threshold(theta0, x):
                t = max(-1 * theta0, min(theta0, x))
                return t

            epsilon = 0.000001  # used to catch DIV0 errors
            verbose = False

            if (strat > 1.0) or (strat < -1.0):
                # out of range
                sys.exit('FAIL: PRZI.getorder() self.strat out of range\n')

            if (dirn != 1.0) and (dirn != -1.0):
                # out of range
                sys.exit('FAIL: PRZI.calc_cdf() bad dirn\n')

            if pmax < pmin:
                # screwed
                sys.exit('FAIL: pmax < pmin\n')

            dxs = dirn * self.strat

            if verbose:
                print('calc_cdf_lut: dirn=%d dxs=%d pmin=%d pmax=%d\n' % (dirn, dxs, pmin, pmax))

            p_range = float(pmax - pmin)
            if p_range < 1:
                # special case: the SHVR-style strategy has shaved all the way to the limit price
                # the lower and upper bounds on the interval are adjacent prices;
                # so cdf is simply the lower price with probability 1

                cdf = [{'price': pmin, 'cum_prob': 1.0}]

                if verbose:
                    print('\n\ncdf:', cdf)

                return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

            c = threshold(t0, m * math.tan(math.pi * (strat + 0.5)))

            # catch div0 errors here
            if abs(c) < epsilon:
                if c > 0:
                    c = epsilon
                else:
                    c = -epsilon

            e2cm1 = math.exp(c) - 1

            # calculate the discrete calligraphic-P function over interval [pmin, pmax]
            # (i.e., this is Equation 8 in the PRZI Technical Note)
            calp_interval = []
            calp_sum = 0
            for p in range(pmin, pmax + 1):
                p_r = (p - pmin) / (p_range)  # p_r in [0.0, 1.0]
                if self.strat == 0.0:
                    # special case: this is just ZIC
                    cal_p = 1 / (p_range + 1)
                elif self.strat > 0:
                    cal_p = (math.exp(c * p_r) - 1.0) / e2cm1
                else:  # self.strat < 0
                    cal_p = 1.0 - ((math.exp(c * p_r) - 1.0) / e2cm1)
                if cal_p < 0:
                    cal_p = 0  # just in case
                calp_interval.append({'price': p, "cal_p": cal_p})
                calp_sum += cal_p

            if calp_sum <= 0:
                print('calp_interval:', calp_interval)
                print('pmin=%f, pmax=%f, calp_sum=%f' % (pmin, pmax, calp_sum))

            cdf = []
            cum_prob = 0
            # now go thru interval summing and normalizing to give the CDF
            for p in range(pmin, pmax + 1):
                price = calp_interval[p - pmin]['price']
                cal_p = calp_interval[p - pmin]['cal_p']
                prob = cal_p / calp_sum
                cum_prob += prob
                cdf.append({'price': p, 'cum_prob': cum_prob})

            if verbose:
                print('\n\ncdf:', cdf)

            return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

        verbose = False

        if verbose:
            print('PRZI getorder: strat=%f' % self.strat)

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            # unpack the assignment-order
            limit = self.orders[0].price
            otype = self.orders[0].otype

            # get extreme limits on price interval
            # lowest price the market will bear
            minprice = int(lob['bids']['worst'])  # default assumption: worst bid price possible is 1 tick
            # trader's individual estimate highest price the market will bear
            maxprice = self.pmax  # default assumption
            if self.pmax is None:
                maxprice = int(limit * self.pmax_c_i + 0.5)  # in the absence of any other info, guess
                self.pmax = maxprice
            elif lob['asks']['sess_hi'] is not None:
                if self.pmax < lob['asks']['sess_hi']:  # some other trader has quoted higher than I expected
                    maxprice = lob['asks']['sess_hi']  # so use that as my new estimate of highest
                    self.pmax = maxprice

            # what price would a SHVR quote?
            p_shvr = shvr_price(otype, limit, lob)

            # it may be more efficient to detect the ZIC special case and generate a price directly
            # whether it is or not depends on how many entries need to be sampled in the LUT reverse-lookup
            # versus the compute time of the call to random.randint that would be used in direct ZIC
            # here, for simplicity, we're not treating ZIC as a special case...
            # ... so the full CDF LUT needs to be instantiated for ZIC (strat=0.0) just like any other strat value

            # use the cdf look-up table
            # cdf_lut is a list of little dictionaries
            # each dictionary has form: {'cum_prob':nnn, 'price':nnn}
            # generate u=U(0,1) uniform disrtibution
            # starting with the lowest nonzero cdf value at cdf_lut[0],
            # walk up the lut (i.e., examine higher cumulative probabilities),
            # until we're in the range of u; then return the relevant price

            # the LUTs are re-computed if any of the details have changed
            if otype == 'Bid':

                # direction * strat
                dxs = -1 * self.strat  # the minus one multiplier is the "buy" direction

                p_max = int(limit)
                if dxs <= 0:
                    p_min = minprice  # this is delta_p for BSE, i.e. ticksize =1
                else:
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    p_min = int(0.5 + (dxs * p_shvr) + ((1.0 - dxs) * minprice))

                if (self.cdf_lut_bid is None) or \
                        (self.cdf_lut_bid['strat'] != self.strat) or \
                        (self.cdf_lut_bid['pmin'] != p_min) or \
                        (self.cdf_lut_bid['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New bid LUT')
                    self.cdf_lut_bid = calc_cdf_lut(self.strat, self.theta0,
                                                    self.m, -1, p_min, p_max)

                lut = self.cdf_lut_bid

            else:  # otype == 'Ask'

                dxs = self.strat

                p_min = int(limit)
                if dxs <= 0:
                    p_max = maxprice
                else:
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (dxs * p_shvr) + ((1.0 - dxs) * maxprice))

                if (self.cdf_lut_ask is None) or \
                        (self.cdf_lut_ask['strat'] != self.strat) or \
                        (self.cdf_lut_ask['pmin'] != p_min) or \
                        (self.cdf_lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New ask LUT')
                    self.cdf_lut_ask = calc_cdf_lut(self.strat, self.theta0,
                                                    self.m, +1, p_min, p_max)

                lut = self.cdf_lut_ask

            if verbose:
                print('PRZI LUT =', lut)

            # do inverse lookup on the LUT to find the price
            u = random.random()
            for entry in lut['cdf_lut']:
                if u < entry['cum_prob']:
                    quoteprice = entry['price']
                    break

            order = Order(self.tid, otype,
                          quoteprice, self.orders[0].qty, time, lob['QID'])

            self.lastquote = order

        return order


# Trader subclass PRZI_SHC (ticker: PRSH)
# added 23 Aug 2021
# Dave Cliff's Parameterized-Response Zero-Intelligence (PRZI) trader
# but with adaptive strategy, as a k-point stochastic hill-climber (SHC) hence PRZI-SHC.
# PRZI-SHC pronounced "prezzy-shuck". Ticker symbol PRSH pronounced "purrsh".

class Trader_PRZI_SHC(Trader):

    # how to mutate the strategy values when hill-climbing
    def mutate_strat(self, s):
        sdev = 0.05
        newstrat = s
        while newstrat == s:
            newstrat = s + random.gauss(0.0, sdev)
            newstrat = max(-1.0, min(1.0, newstrat))
        return newstrat

    def strat_str(self):
        # pretty-print a string summarising this trader's strategies
        string = 'PRSH: %s active_strat=[%d]:\n' % (self.tid, self.active_strat)
        for s in range(0, self.k):
            strat = self.strats[s]
            stratstr = '[%d]: s=%f, start=%f, $=%f, pps=%f\n' % \
                       (s, strat['stratval'], strat['start_t'], strat['profit'], strat['pps'])
            string = string + stratstr

        return string

    def __init__(self, ttype, tid, balance, time):
        # PRZI strategy defined by parameter "strat"
        # here this is randomly assigned
        # strat * direction = -1 = > GVWY; =0 = > ZIC; =+1 = > SHVR

        verbose = False

        Trader.__init__(self, ttype, tid, balance, time)
        self.theta0 = 100  # threshold-function limit value
        self.m = 4  # tangent-function multiplier
        self.k = 4  # number of hill-climbing points (cf number of arms on a multi-armed-bandit)
        self.strat_wait_time = 900  # how many secs do we give any one strat before switching?
        self.strat_range_min = -1.0  # lower-bound on randomly-assigned strategy-value
        self.strat_range_max = +1.0  # upper-bound on randomly-assigned strategy-value
        self.active_strat = 0  # which of the k strategies are we currently playing? -- start with 0
        self.prev_qid = None  # previous order i.d.
        self.strat_eval_time = self.k * self.strat_wait_time  # time to cycle through evaluating all k strategies
        self.last_strat_change_time = time  # what time did we last change strategies?
        self.profit_epsilon = 0.01 * random.random()  # minimum profit-per-sec difference between strategies that counts
        self.strats = []  # strategies awaiting initialization
        self.pmax = None  # this trader's estimate of the maximum price the market will bear
        self.pmax_c_i = math.sqrt(random.randint(1, 10))  # multiplier coefficient when estimating p_max

        for s in range(0, self.k):
            # initialise each of the strategies in sequence
            start_time = time
            profit = 0.0
            profit_per_second = 0
            lut_bid = None
            lut_ask = None
            if s == 0:
                strategy = random.uniform(self.strat_range_min, self.strat_range_max)
            else:
                strategy = self.mutate_strat(self.strats[0]['stratval'])  # mutant of strats[0]
            self.strats.append({'stratval': strategy, 'start_t': start_time,
                                'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})

        if verbose:
            print("PRSH %s %s\n" % (tid, self.strat_str()))

    def getorder(self, time, countdown, lob):

        # shvr_price tells us what price a SHVR would quote in these circs
        def shvr_price(otype, limit, lob):

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    shvr_p = lob['bids']['best'] + 1  # BSE tick size is always 1
                    if shvr_p > limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    shvr_p = lob['asks']['best'] - 1  # BSE tick size is always 1
                    if shvr_p < limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['asks']['worst']

            return shvr_p

        # calculate cumulative distribution function (CDF) look-up table (LUT)
        def calc_cdf_lut(strat, t0, m, dirn, pmin, pmax):
            # set parameter values and calculate CDF LUT
            # dirn is direction: -1 for buy, +1 for sell

            # the threshold function used to clip
            def threshold(theta0, x):
                t = max(-1 * theta0, min(theta0, x))
                return t

            epsilon = 0.000001  # used to catch DIV0 errors
            verbose = False

            if (strat > 1.0) or (strat < -1.0):
                # out of range
                sys.exit('PRSH FAIL: PRZI.getorder() self.strat out of range\n')

            if (dirn != 1.0) and (dirn != -1.0):
                # out of range
                sys.exit('PRSH FAIL: PRZI.calc_cdf() bad dirn\n')

            if pmax < pmin:
                # screwed
                sys.exit('PRSH FAIL: pmax < pmin\n')

            dxs = dirn * strat

            if verbose:
                print('PRSH calc_cdf_lut: dirn=%d dxs=%d pmin=%d pmax=%d\n' % (dirn, dxs, pmin, pmax))

            p_range = float(pmax - pmin)
            if p_range < 1:
                # special case: the SHVR-style strategy has shaved all the way to the limit price
                # the lower and upper bounds on the interval are adjacent prices;
                # so cdf is simply the lower price with probability 1

                cdf = [{'price': pmin, 'cum_prob': 1.0}]

                if verbose:
                    print('\n\ncdf:', cdf)

                return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

            c = threshold(t0, m * math.tan(math.pi * (strat + 0.5)))

            # catch div0 errors here
            if abs(c) < epsilon:
                if c > 0:
                    c = epsilon
                else:
                    c = -epsilon

            e2cm1 = math.exp(c) - 1

            # calculate the discrete calligraphic-P function over interval [pmin, pmax]
            # (i.e., this is Equation 8 in the PRZI Technical Note)
            calp_interval = []
            calp_sum = 0
            for p in range(pmin, pmax + 1):
                p_r = (p - pmin) / (p_range)  # p_r in [0.0, 1.0]
                if strat == 0.0:
                    # special case: this is just ZIC
                    cal_p = 1 / (p_range + 1)
                elif strat > 0:
                    cal_p = (math.exp(c * p_r) - 1.0) / e2cm1
                else:  # self.strat < 0
                    cal_p = 1.0 - ((math.exp(c * p_r) - 1.0) / e2cm1)
                if cal_p < 0:
                    cal_p = 0  # just in case
                calp_interval.append({'price': p, "cal_p": cal_p})
                calp_sum += cal_p

            if calp_sum <= 0:
                print('calp_interval:', calp_interval)
                print('pmin=%f, pmax=%f, calp_sum=%f' % (pmin, pmax, calp_sum))

            cdf = []
            cum_prob = 0
            # now go thru interval summing and normalizing to give the CDF
            for p in range(pmin, pmax + 1):
                price = calp_interval[p - pmin]['price']
                cal_p = calp_interval[p - pmin]['cal_p']
                prob = cal_p / calp_sum
                cum_prob += prob
                cdf.append({'price': p, 'cum_prob': cum_prob})

            if verbose:
                print('\n\ncdf:', cdf)

            return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

        verbose = False

        if verbose:
            print('t=%f PRSH getorder: %s, %s' % (time, self.tid, self.strat_str()))

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            # unpack the assignment-order
            limit = self.orders[0].price
            otype = self.orders[0].otype
            qid = self.orders[0].qid

            if self.prev_qid is None:
                self.prev_qid = qid

            if qid != self.prev_qid:
                # customer-order i.d. has changed, so we're working a new customer-order now
                # this is the time to switch arms
                # print("New order! (how does it feel?)")
                dummy = 1

            # get extreme limits on price interval
            # lowest price the market will bear
            minprice = int(lob['bids']['worst'])  # default assumption: worst bid price possible is 1 tick
            # trader's individual estimate highest price the market will bear
            maxprice = self.pmax  # default assumption
            if self.pmax is None:
                maxprice = int(limit * self.pmax_c_i + 0.5)  # in the absence of any other info, guess
                self.pmax = maxprice
            elif lob['asks']['sess_hi'] is not None:
                if self.pmax < lob['asks']['sess_hi']:  # some other trader has quoted higher than I expected
                    maxprice = lob['asks']['sess_hi']  # so use that as my new estimate of highest
                    self.pmax = maxprice

            # what price would a SHVR quote?
            p_shvr = shvr_price(otype, limit, lob)

            # it may be more efficient to detect the ZIC special case and generate a price directly
            # whether it is or not depends on how many entries need to be sampled in the LUT reverse-lookup
            # versus the compute time of the call to random.randint that would be used in direct ZIC
            # here, for simplicity, we're not treating ZIC as a special case...
            # ... so the full CDF LUT needs to be instantiated for ZIC (strat=0.0) just like any other strat value

            # use the cdf look-up table
            # cdf_lut is a list of little dictionaries
            # each dictionary has form: {'cum_prob':nnn, 'price':nnn}
            # generate u=U(0,1) uniform disrtibution
            # starting with the lowest nonzero cdf value at cdf_lut[0],
            # walk up the lut (i.e., examine higher cumulative probabilities),
            # until we're in the range of u; then return the relevant price

            strat = self.strats[self.active_strat]['stratval']

            if otype == 'Bid':

                # direction * strat
                dxs = -1 * strat  # the minus one multiplier is the "buy" direction

                p_max = int(limit)
                if dxs <= 0:
                    p_min = minprice  # this is delta_p for BSE, i.e. ticksize =1
                else:
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    p_min = int(0.5 + (dxs * p_shvr) + ((1.0 - dxs) * minprice))

                lut_bid = self.strats[self.active_strat]['lut_bid']
                if (lut_bid is None) or \
                        (lut_bid['strat'] != strat) or \
                        (lut_bid['pmin'] != p_min) or \
                        (lut_bid['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New bid LUT')
                    self.strats[self.active_strat]['lut_bid'] = calc_cdf_lut(strat, self.theta0, self.m, -1, p_min,
                                                                             p_max)

                lut = self.strats[self.active_strat]['lut_bid']

            else:  # otype == 'Ask'

                dxs = strat

                p_min = int(limit)
                if dxs <= 0:
                    p_max = maxprice
                else:
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (dxs * p_shvr) + ((1.0 - dxs) * maxprice))

                lut_ask = self.strats[self.active_strat]['lut_ask']
                if (lut_ask is None) or \
                        (lut_ask['strat'] != strat) or \
                        (lut_ask['pmin'] != p_min) or \
                        (lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New ask LUT')
                    self.strats[self.active_strat]['lut_ask'] = calc_cdf_lut(strat, self.theta0, self.m, +1, p_min,
                                                                             p_max)

                lut = self.strats[self.active_strat]['lut_ask']

            if verbose:
                # print('PRZI LUT =', lut)
                # print ('[LUT print suppressed]')
                dummy = 1

            # do inverse lookup on the LUT to find the price
            u = random.random()
            for entry in lut['cdf_lut']:
                if u < entry['cum_prob']:
                    quoteprice = entry['price']
                    break

            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])

            self.lastquote = order

        return order

    def bookkeep(self, trade, order, verbose, time):

        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        self.blotter = self.blotter[-self.blotter_length:]  # right-truncate to keep to length

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            sys.exit('PRSH FAIL: negative profit')

        if verbose: print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
        self.del_order(order)  # delete the order

        # Trader.bookkeep(self, trade, order, verbose, time)

        # Check: bookkeep is only called after a successful trade? i.e. no need to check re trade or not

        # so ...
        # if I have just traded and I am a PRSH trader
        # then I want to reset the timer on the current strat and update its profit sum

        self.strats[self.active_strat]['profit'] += profit

    # PRSH respond() asks/answers two questions
    # do we need to choose a new strategy? (i.e. have just completed/cancelled previous customer order)
    # do we need to dump one arm and generate a new one? (i.e., both/all arms have been evaluated enough)
    def respond(self, time, lob, trade, verbose):

        shc_algo = 'basic'

        # "basic" is a very basic form of stochastic hill-cliber (SHC) that v easy to understand and to code
        # it cycles through the k different strats until each has been operated for at least eval_time seconds
        # but a strat that does nothing will get swapped out if it's been running for no_deal_time without a deal
        # then the strats with the higher total accumulated profit is retained,
        # and mutated versions of it are copied into the other strats
        # then all counters are reset, and this is repeated indefinitely
        # e.g. inspired by multi-arm-bandit algos like like epsilon-greedy, softmax, or upper confidence bound (UCB)

        verbose = False

        # first update each strategy's profit-per-second value -- this is the "fitness" of each strategy
        for s in self.strats:
            pps_time = time - s['start_t']
            if pps_time > 0:
                s['pps'] = s['profit'] / pps_time
            else:
                s['pps'] = 0.0

        if shc_algo == 'basic':

            if verbose:
                # print('t=%f %s PRSH respond: shc_algo=%s eval_t=%f max_wait_t=%f' %
                #     (time, self.tid, shc_algo, self.strat_eval_time, self.strat_wait_time))
                dummy = 1

            # do we need to swap strategies?
            # this is based on time elapsed since last reset -- waiting for the current strategy to get a deal
            # -- otherwise a hopeless strategy can just sit there for ages doing nothing,
            # which would disadvantage the *other* strategies because they would never get a chance to score any profit.
            # when a trader does a deal, clock is reset;
            # clock also reset when new a strat is created, obvs.

            # NB this *cycles* through the available strats in sequence

            s = self.active_strat
            time_elapsed = time - self.last_strat_change_time
            if time_elapsed > self.strat_wait_time:
                # we have waited long enough: swap to another strategy

                new_strat = s + 1
                if new_strat > self.k - 1:
                    new_strat = 0

                self.active_strat = new_strat
                self.last_strat_change_time = time

                if verbose:
                    print('t=%f %s PRSH respond: strat[%d] elapsed=%f; wait_t=%f, switched to strat=%d' %
                          (time, self.tid, s, time_elapsed, self.strat_wait_time, new_strat))

            # code below here deals with creating a new set of k-1 mutants from the best of the k strats

            for s in self.strats:
                # assume that all strats have had long enough, and search for evidence to the contrary
                all_old_enough = True
                lifetime = time - s['start_t']
                if lifetime < self.strat_eval_time:
                    all_old_enough = False
                    break

            if all_old_enough:
                # all strategies have had long enough: which has made most profit?

                # sort them by profit
                strats_sorted = sorted(self.strats, key=lambda k: k['pps'], reverse=True)
                # strats_sorted = self.strats     # use this as a control: unsorts the strats, gives pure random walk.

                if verbose:
                    print('PRSH %s: strat_eval_time=%f, all_old_enough=True' % (self.tid, self.strat_eval_time))
                    for s in strats_sorted:
                        print('s=%f, start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time - s['start_t'], s['profit'], s['pps']))

                # if the difference between the top two strats is too close to call then flip a coin
                # this is to prevent the same good strat being held constant simply by chance cos it is at index [0]
                prof_diff = strats_sorted[0]['profit'] - strats_sorted[1]['profit']
                if abs(prof_diff) < self.profit_epsilon:
                    # they're too close to call, so just flip a coin
                    best_strat = random.randint(0, 1)
                elif prof_diff > 0:
                    best_strat = 0
                else:
                    best_strat = 1

                if best_strat == 1:
                    # need to swap strats[0] and strats[1]
                    tmp_strat = strats_sorted[0]
                    strats_sorted[0] = strats_sorted[1]
                    strats_sorted[1] = tmp_strat

                # the sorted list of strats replaces the existing list
                self.strats = strats_sorted

                # at this stage, strats_sorted[0] is our newly-chosen elite-strat, about to replicate
                # record it

                # now replicate and mutate elite into all the other strats
                for s in range(1, self.k):  # note range index starts at one not zero
                    self.strats[s]['stratval'] = self.mutate_strat(self.strats[0]['stratval'])
                    self.strats[s]['start_t'] = time
                    self.strats[s]['profit'] = 0.0
                    self.strats[s]['pps'] = 0.0
                # and then update (wipe) records for the elite
                self.strats[0]['start_t'] = time
                self.strats[0]['profit'] = 0.0
                self.strats[0]['pps'] = 0.0

                if verbose:
                    print('%s: strat_eval_time=%f, MUTATED:' % (self.tid, self.strat_eval_time))
                    for s in self.strats:
                        print('s=%f start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time - s['start_t'], s['profit'], s['pps']))

        else:
            sys.exit('FAIL: bad value for shc_algo')


class Trader_xgboost(Trader):
    def __init__(self, ttype, tid, balance, time):
        Trader.__init__(self, ttype, tid, balance, time)
        self.limit = None
        self.job = None
        self.smithsAlpha = None
        self.lastTrades = []
        self.smithsAlphaMin = None
        self.smithsAlphaMax = None
        self.eqlbm = None
        self.nLastTrades = 5  # N in AIJ08
        self.ema_param = 2 / float(nLastTrades + 1)
        self.prev_time = 0

        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

    def updateEq(self, price):
        # Updates the equilibrium price estimate using EMA
        # global eqlbm
        if self.eqlbm == None:
            self.eqlbm = price
        else:
            self.eqlbm = self.ema_param * price + (1 - self.ema_param) * self.eqlbm
        return self.eqlbm

    def updateSmithsAlpha(self, price):
        self.lastTrades.append(price)
        if not (len(self.lastTrades) <= self.nLastTrades): self.lastTrades.pop(0)
        self.smithsAlpha = math.sqrt(
            sum(((p - self.eqlbm) ** 2) for p in self.lastTrades) * (1 / float(len(self.lastTrades)))) / self.eqlbm
        return self.smithsAlpha

    def getorder(self, time, countdown, lob):
        output_flag = False
        if len(self.orders) < 1:
            order = None
            # return order
        else:
            self.limit = self.orders[0].price
            self.job = self.orders[0].otype
            if self.job == 'Bid':
                if lob['asks']['best'] is not None:
                    if self.limit >= lob['asks']['best']:
                        output_flag = True
                    else:
                        output_flag = False
                else:
                    output_flag = False
            else:
                if lob['bids']['best'] is not None:
                    if self.limit <= lob['bids']['best']:
                        output_flag = True
                    else:
                        output_flag = False
                else:
                    output_flag = False

            output_time = time
            output_price = self.orders[0].price
            if lob['asks']['best'] and lob['bids']['best']:
                output_spread = lob['asks']['best'] - lob['bids']['best']
            else:
                output_spread = 0
            output_midprice = lob['midprice']
            output_microprice = lob['microprice']
            if lob['bids']['best'] is None:
                output_best_bids = 0
            else:
                output_best_bids = lob['bids']['best']
            if lob['asks']['best'] is None:
                output_best_asks = 0
            else:
                output_best_asks = lob['asks']['best']
            if output_midprice and output_microprice:
                output_imbalance = output_microprice - output_midprice
            else:
                output_imbalance = 0
            output_quotes_quantity = lob['bids']['n'] + lob['bids']['n']
            output_eq = updateEq(output_price)
            output_smithsAlpha = updateSmithsAlpha(output_price)
            if self.prev_time > 0:
                output_time_elapsed = time - self.prev_time
            else:
                output_time_elapsed = time
            # lst = ['Time':[output_time]]
            data = {'Time': [output_time], 'flag': [output_flag], 'order price': [output_price],
                    'spread': [output_spread], 'midprice': [output_midprice], 'microprice': [output_microprice],
                    'best_bids': [output_best_bids], 'best_asks': [output_best_asks],
                    'time_elapsed': [output_time_elapsed], 'imbalance': [output_imbalance],
                    'quotes_quantity': [output_quotes_quantity], 'equilibrium price': [output_eq],
                    'smithsAlpha': [output_smithsAlpha]}
            loaded_data = pd.DataFrame(data)
            # print(loaded_data)

            loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
            quoteprice = loaded_model.predict(loaded_data)[0]
            '''
            print('best bids')
            print(output_best_bids)
            print('best asks')
            print(output_best_asks)
            print('order price')
            print(output_price)
            print('trade')
            print(quoteprice)
            print(self.job)'''

            if self.job =='Ask' and quoteprice < output_price:
                quoteprice = output_price
            if self.job =='Bid' and quoteprice > output_price:
                quoteprice = output_price
            order = Order(self.tid, self.job, quoteprice, self.orders[0].qty, time, lob['QID'])
        return order


    def respond(self, time, lob, trade, verbose):
        bid_improved = False
        bid_hit = False
        lob_best_bid_p = lob['bids']['best']
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = lob['bids']['lob'][-1][1]
            if (self.prev_best_bid_p is not None) and (self.prev_best_bid_p < lob_best_bid_p):
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        lob_best_ask_p = lob['asks']['best']
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = lob['asks']['lob'][0][1]
            if (self.prev_best_ask_p is not None) and (self.prev_best_ask_p > lob_best_ask_p):
                # best ask has improved -- NB doesn't check if the improvement was by self
                ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # -- assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                ask_lifted = False
            else:
                ask_lifted = True

        if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
            print('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)

        deal = bid_hit or ask_lifted
        if deal:
            self.prev_time = time
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q


class Trader_AA(Trader):

    def __init__(self, ttype, tid, balance, time):
        Trader.__init__(self, ttype, tid, balance, time)

        self.limit = None
        self.job = None

        # learning variables
        self.r_shout_change_relative = 0.05
        self.r_shout_change_absolute = 0.05
        self.short_term_learning_rate = random.uniform(0.1, 0.5)
        self.long_term_learning_rate = random.uniform(0.1, 0.5)
        self.moving_average_weight_decay = 0.95  # how fast weight decays with time, lower is quicker, 0.9 in vytelingum
        self.moving_average_window_size = 5
        self.offer_change_rate = 3.0
        self.theta = -2.0
        self.theta_max = 2.0
        self.theta_min = -8.0
        self.marketMax = bse_sys_maxprice

        # Variables to describe the market
        self.previous_transactions = []
        self.moving_average_weights = []
        for i in range(self.moving_average_window_size):
            self.moving_average_weights.append(self.moving_average_weight_decay ** i)
        self.estimated_equilibrium = []
        self.smiths_alpha = []
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

        # Trading Variables
        self.r_shout = None
        self.buy_target = None
        self.sell_target = None
        self.buy_r = -1.0 * (0.3 * random.random())
        self.sell_r = -1.0 * (0.3 * random.random())

    def calcEq(self):  ##clear and correct
        # Slightly modified from paper, it is unclear inpaper
        # N previous transactions * weights / N in vytelingum, swap N denominator for sum of weights to be correct?
        if len(self.previous_transactions) == 0:
            return
        elif len(self.previous_transactions) < self.moving_average_window_size:
            # Not enough transactions
            self.estimated_equilibrium.append(
                float(sum(self.previous_transactions)) / max(len(self.previous_transactions), 1))
        else:
            N_previous_transactions = self.previous_transactions[-self.moving_average_window_size:]
            thing = [N_previous_transactions[i] * self.moving_average_weights[i] for i in
                     range(self.moving_average_window_size)]
            eq = sum(thing) / sum(self.moving_average_weights)
            self.estimated_equilibrium.append(eq)

    def calcAlpha(self):  ##correct. but calcAlpha in snashall's version is incorrect
        alpha = 0.0
        for p in self.previous_transactions:
            alpha += (p - self.estimated_equilibrium[-1]) ** 2
        alpha = math.sqrt(alpha / len(self.previous_transactions))
        self.smiths_alpha.append(alpha / self.estimated_equilibrium[-1])

    def calcTheta(self):  ## clear and correct
        gamma = 2.0  # not sensitive apparently so choose to be whatever
        # necessary for intialisation, div by 0
        if min(self.smiths_alpha) == max(self.smiths_alpha):
            alpha_range = 0.4  # starting value i guess
        else:
            alpha_range = (self.smiths_alpha[-1] - min(self.smiths_alpha)) / (
                    max(self.smiths_alpha) - min(self.smiths_alpha))
        theta_range = self.theta_max - self.theta_min
        desired_theta = self.theta_min + (theta_range) * (1 - alpha_range) * math.exp(gamma * (alpha_range - 1))
        self.theta = self.theta + self.long_term_learning_rate * (desired_theta - self.theta)
        if self.theta > self.theta_max:
            self.theta = self.theta_max
        if self.theta < self.theta_min:
            self.theta = self.theta_min

    def calcRshout(self):  ## unclear in Vytelingum's paper
        p = self.estimated_equilibrium[-1]
        l = self.limit
        theta = self.theta
        if self.job == 'Bid':
            # Currently a buyer
            if l <= p:  # extramarginal!
                self.r_shout = 0.0
            else:  # intramarginal :(
                if self.buy_target > self.estimated_equilibrium[-1]:
                    # r[0,1]
                    self.r_shout = math.log(((self.buy_target - p) * (math.exp(theta) - 1) / (l - p)) + 1) / theta
                else:
                    # r[-1,0]
                    # print 'buy_target: %f , p: %f , theta: %f' %(self.buy_target,p,theta)
                    self.r_shout = math.log((1 - (self.buy_target / p)) * (math.exp(theta) - 1) + 1) / theta
                # self.r_shout = self.buy_r

        if self.job == 'Ask':
            # Currently a seller
            if l >= p:  # extramarginal!
                self.r_shout = 0
            else:  # intramarginal :(
                if self.sell_target > self.estimated_equilibrium[-1]:
                    # r[-1,0]
                    self.r_shout = math.log(
                        (self.sell_target - p) * (math.exp(theta) - 1) / (self.marketMax - p) + 1) / theta
                else:
                    # r[0,1]
                    a = (self.sell_target - l) / (p - l)
                    self.r_shout = (math.log((1 - a) * (math.exp(theta) - 1) + 1)) / theta
                # self.r_shout = self.sell_r

    def calcAgg(self):
        delta = 0
        if self.job == 'Bid':
            # BUYER
            if self.buy_target >= self.previous_transactions[-1]:
                # must be more aggressive
                delta = (1 + self.r_shout_change_relative) * self.r_shout + self.r_shout_change_absolute
            else:
                delta = (1 - self.r_shout_change_relative) * self.r_shout - self.r_shout_change_absolute

            self.buy_r = self.buy_r + self.short_term_learning_rate * (delta - self.buy_r)

        if self.job == 'Ask':
            # SELLER
            if self.sell_target > self.previous_transactions[-1]:
                delta = (1 + self.r_shout_change_relative) * self.r_shout + self.r_shout_change_absolute
            else:
                delta = (1 - self.r_shout_change_relative) * self.r_shout - self.r_shout_change_absolute

            self.sell_r = self.sell_r + self.short_term_learning_rate * (delta - self.sell_r)

    def calcTarget(self):
        if len(self.estimated_equilibrium) > 0:
            p = self.estimated_equilibrium[-1]
            if self.limit == p:
                p = p * 1.000001  # to prevent theta_bar = 0
        elif self.job == 'Bid':
            p = self.limit - self.limit * 0.2  ## Initial guess for eq if no deals yet!!....
        elif self.job == 'Ask':
            p = self.limit + self.limit * 0.2
        l = self.limit
        theta = self.theta
        if self.job == 'Bid':
            # BUYER
            minus_thing = self.buy_r * math.exp(theta * (self.buy_r - 1))

            if l <= p:  # Extramarginal
                if self.buy_r >= 0:
                    self.buy_target = l
                else:
                    self.buy_target = l * (1 - minus_thing)
            else:  # intramarginal
                if self.buy_r >= 0:
                    # theta_ba = (p * math.exp(-theta))/(l-p)-1
                    theta_ba = theta
                    # print 'theta: %f' %(self.theta)
                    # print 'theta_ba: %f '%(theta_ba)
                    # print 'l-p: %f '%(l-p)
                    # print 'self.buy_r :%f' %(self.buy_r)

                    self.buy_target = (l - p) * (1 - (self.buy_r + 1) * math.exp(self.buy_r * theta_ba)) + p
                else:
                    self.buy_target = p * (1 - minus_thing)
            if self.buy_target > l:
                self.buy_target = l
            if self.buy_target < bse_sys_minprice:
                self.buy_target = bse_sys_minprice
            # print 'buy_target = %f'%(self.buy_target)

        if self.job == 'Ask':
            # SELLER

            if l <= p:  # Intramarginal
                if self.buy_r >= 0:
                    self.buy_target = p + (p - l) * self.sell_r * math.exp((self.sell_r - 1) * theta)
                else:
                    theta_ba = math.log((self.marketMax - p) / (p - l)) - theta
                    self.buy_target = p + (self.marketMax - p) * self.sell_r * math.exp((self.sell_r + 1) * theta_ba)
            else:  # Extramarginal
                if self.buy_r >= 0:
                    self.buy_target = l
                else:
                    self.buy_target = l + (self.marketMax - l) * self.sell_r * math.exp((self.sell_r - 1) * theta)
            if (self.sell_target is not None) and (self.sell_target < l):
                self.sell_target = l
            if (self.sell_target is not None) and (self.sell_target > bse_sys_maxprice):
                self.sell_target = bse_sys_maxprice
            # print 'sell_target = %f'%(self.sell_target)

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            self.active = False
            return None
        else:
            self.active = True
            self.limit = self.orders[0].price
            self.job = self.orders[0].otype
            self.calcTarget()

            if self.prev_best_bid_p == None:
                o_bid = 0
            else:
                o_bid = self.prev_best_bid_p
            if self.prev_best_ask_p == None:
                o_ask = self.marketMax
            else:
                o_ask = self.prev_best_ask_p

            if self.job == 'Bid':  # BUYER
                if self.limit <= o_bid:
                    return None
                else:
                    if len(self.previous_transactions) <= 0:  ## has been at least one transaction
                        o_ask_plus = (1 + self.r_shout_change_relative) * o_ask + self.r_shout_change_absolute
                        quoteprice = o_bid + ((min(self.limit, o_ask_plus) - o_bid) / self.offer_change_rate)
                    else:
                        if o_ask <= self.buy_target:
                            quoteprice = o_ask
                        else:
                            quoteprice = o_bid + ((self.buy_target - o_bid) / self.offer_change_rate)
            if self.job == 'Ask':
                if self.limit >= o_ask:
                    return None
                else:
                    if len(self.previous_transactions) <= 0:  ## has been at least one transaction
                        o_bid_minus = (1 - self.r_shout_change_relative) * o_bid - self.r_shout_change_absolute
                        quoteprice = o_ask - ((o_ask - max(self.limit, o_bid_minus)) / self.offer_change_rate)
                    else:
                        if o_bid >= self.sell_target:
                            quoteprice = o_bid
                        else:
                            quoteprice = o_ask - ((o_ask - self.sell_target) / self.offer_change_rate)

            order = Order(self.tid, self.job, quoteprice, self.orders[0].qty, time, lob['QID'])
        return order

    def respond(self, time, lob, trade, verbose):
        ## Begin nicked from ZIP

        # what, if anything, has happened on the bid LOB? Nicked from ZIP..
        bid_improved = False
        bid_hit = False
        lob_best_bid_p = lob['bids']['best']
        lob_best_bid_q = None
        if lob_best_bid_p != None:
            # non-empty bid LOB
            lob_best_bid_q = lob['bids']['lob'][0][1]
            if (self.prev_best_bid_p is not None) and (self.prev_best_bid_p < lob_best_bid_p):
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif trade != None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p != None:
            # # the bid LOB has been emptied: was it cancelled or hit?
            # last_tape_item = lob['tape'][-1]
            # if last_tape_item['type'] == 'Cancel' :
            #         bid_hit = False
            # else:
            #         bid_hit = True
            # the bid LOB is empty now but was not previously: so was it canceled or lifted?
            if trade != None:
                # a trade has occurred and the previously nonempty ask LOB is now empty
                # so assume best ask was lifted
                bid_hit = True
            else:
                bid_hit = False
        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        lob_best_ask_p = lob['asks']['best']
        lob_best_ask_q = None
        if lob_best_ask_p != None:
            # non-empty ask LOB
            lob_best_ask_q = lob['asks']['lob'][0][1]
            if (self.prev_best_ask_p is not None) and (self.prev_best_ask_p > lob_best_ask_p):
                # best ask has improved -- NB doesn't check if the improvement was by self
                ask_improved = True
            elif trade != None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced -- assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p != None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            # last_tape_item = lob['tape'][-1]
            # if last_tape_item['type'] == 'Cancel' :
            #         ask_lifted = False
            # else:
            #         ask_lifted = True
            # the ask LOB is empty now but was not previously: so was it canceled or lifted?
            if trade != None:
                # a trade has occurred and the previously nonempty ask LOB is now empty
                # so assume best ask was lifted
                ask_lifted = True
            else:
                ask_lifted = False

        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q

        deal = bid_hit or ask_lifted

        ## End nicked from ZIP

        if deal:
            self.previous_transactions.append(trade['price'])
            if self.sell_target == None:
                self.sell_target = trade['price']
            if self.buy_target == None:
                self.buy_target = trade['price']
            self.calcEq()
            self.calcAlpha()
            self.calcTheta()
            self.calcRshout()
            self.calcAgg()
            self.calcTarget()


# Trader subclass ZIP
# After Cliff 1997
class Trader_ZIP(Trader):

    # ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
    # NB this implementation keeps separate margin values for buying & selling,
    #    so a single trader can both buy AND sell
    #    -- in the original, traders were either buyers OR sellers

    def __init__(self, ttype, tid, balance, time):
        Trader.__init__(self, ttype, tid, balance, time)
        self.willing = 1
        self.able = 1
        self.job = None  # this gets switched to 'Bid' or 'Ask' depending on order-type
        self.active = False  # gets switched to True while actively working an order
        self.prev_change = 0  # this was called last_d in Cliff'97
        self.beta = 0.1 + 0.4 * random.random()
        self.momntm = 0.1 * random.random()
        self.ca = 0.05  # self.ca & .cr were hard-coded in '97 but parameterised later
        self.cr = 0.05
        self.margin = None  # this was called profit in Cliff'97
        self.margin_buy = -1.0 * (0.05 + 0.3 * random.random())
        self.margin_sell = 0.05 + 0.3 * random.random()
        self.price = None
        self.limit = None
        # memory of best price & quantity of best bid and ask, on LOB on previous update
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

    def getorder(self, time, countdown, lob):

        if len(self.orders) < 1:
            self.active = False
            order = None
        else:
            self.active = True
            self.limit = self.orders[0].price
            self.job = self.orders[0].otype
            if self.job == 'Bid':
                # currently a buyer (working a bid order)
                self.margin = self.margin_buy
            else:
                # currently a seller (working a sell order)
                self.margin = self.margin_sell
            quoteprice = int(self.limit * (1 + self.margin))
            self.price = quoteprice

            order = Order(self.tid, self.job, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order

    # update margin on basis of what happened in market
    def respond(self, time, lob, trade, verbose):
        # ZIP trader responds to market events, altering its margin
        # does this whether it currently has an order to work or not

        def target_up(price):
            # generate a higher target price by randomly perturbing given price
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel + ptrb_abs, 0))
            # #                        print('TargetUp: %d %d\n' % (price,target))
            return target

        def target_down(price):
            # generate a lower target price by randomly perturbing given price
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel - ptrb_abs, 0))
            # #                        print('TargetDn: %d %d\n' % (price,target))
            return target

        def willing_to_trade(price):
            # am I willing to trade at this price?
            willing = False
            if self.job == 'Bid' and self.active and self.price >= price:
                willing = True
            if self.job == 'Ask' and self.active and self.price <= price:
                willing = True
            return willing

        def profit_alter(price):
            oldprice = self.price
            diff = price - oldprice
            change = ((1.0 - self.momntm) * (self.beta * diff)) + (self.momntm * self.prev_change)
            self.prev_change = change
            newmargin = ((self.price + change) / self.limit) - 1.0

            if self.job == 'Bid':
                if newmargin < 0.0:
                    self.margin_buy = newmargin
                    self.margin = newmargin
            else:
                if newmargin > 0.0:
                    self.margin_sell = newmargin
                    self.margin = newmargin

            # set the price from limit and profit-margin
            self.price = int(round(self.limit * (1.0 + self.margin), 0))

        # #                        print('old=%d diff=%d change=%d price = %d\n' % (oldprice, diff, change, self.price))

        # what, if anything, has happened on the bid LOB?
        bid_improved = False
        bid_hit = False
        lob_best_bid_p = lob['bids']['best']
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = lob['bids']['lob'][-1][1]
            if (self.prev_best_bid_p is not None) and (self.prev_best_bid_p < lob_best_bid_p):
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        lob_best_ask_p = lob['asks']['best']
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = lob['asks']['lob'][0][1]
            if (self.prev_best_ask_p is not None) and (self.prev_best_ask_p > lob_best_ask_p):
                # best ask has improved -- NB doesn't check if the improvement was by self
                ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # -- assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                ask_lifted = False
            else:
                ask_lifted = True

        if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
            print('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)

        deal = bid_hit or ask_lifted

        if self.job == 'Ask':
            # seller
            if deal:
                tradeprice = trade['price']
                if self.price <= tradeprice:
                    # could sell for more? raise margin
                    target_price = target_up(tradeprice)
                    profit_alter(target_price)
                elif ask_lifted and self.active and not willing_to_trade(tradeprice):
                    # wouldnt have got this deal, still working order, so reduce margin
                    target_price = target_down(tradeprice)
                    profit_alter(target_price)
            else:
                # no deal: aim for a target price higher than best bid
                if ask_improved and self.price > lob_best_ask_p:
                    if lob_best_bid_p is not None:
                        target_price = target_up(lob_best_bid_p)
                    else:
                        target_price = lob['asks']['worst']  # stub quote
                    profit_alter(target_price)

        if self.job == 'Bid':
            # buyer
            if deal:
                tradeprice = trade['price']
                if self.price >= tradeprice:
                    # could buy for less? raise margin (i.e. cut the price)
                    target_price = target_down(tradeprice)
                    profit_alter(target_price)
                elif bid_hit and self.active and not willing_to_trade(tradeprice):
                    # wouldnt have got this deal, still working order, so reduce margin
                    target_price = target_up(tradeprice)
                    profit_alter(target_price)
            else:
                # no deal: aim for target price lower than best ask
                if bid_improved and self.price < lob_best_bid_p:
                    if lob_best_ask_p is not None:
                        target_price = target_down(lob_best_ask_p)
                    else:
                        target_price = lob['bids']['worst']  # stub quote
                    profit_alter(target_price)

        # remember the best LOB data ready for next response
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q


##########################---Below lies the experiment/test-rig---##################


# trade_stats()
# dump CSV statistics on exchange data and trader population to file for later analysis
# this makes no assumptions about the number of types of traders, or
# the number of traders of any one type -- allows either/both to change
# between successive calls, but that does make it inefficient as it has to
# re-analyse the entire set of traders on each call
def trade_stats(expid, traders, dumpfile, time, lob):
    # Analyse the set of traders, to see what types we have
    trader_types = {}
    for t in traders:
        ttype = traders[t].ttype
        if ttype in trader_types.keys():
            t_balance = trader_types[ttype]['balance_sum'] + traders[t].balance
            n = trader_types[ttype]['n'] + 1
        else:
            t_balance = traders[t].balance
            n = 1
        trader_types[ttype] = {'n': n, 'balance_sum': t_balance}

    # first two columns of output are the session_id and the time
    dumpfile.write('%s, %s, ' % (expid, time))

    # second two columns of output are the LOB best bid and best offer (or 'None' if they're undefined)
    if lob['bids']['best'] is not None:
        dumpfile.write('%s, ' % (lob['bids']['best']))
    else:
        dumpfile.write('None, ')
    if lob['asks']['best'] is not None:
        dumpfile.write('%s, ' % (lob['asks']['best']))
    else:
        dumpfile.write('None, ')

    # total remaining number of columns printed depends on number of different trader-types at this timestep
    # for each trader type we print FOUR columns...
    # TraderTypeCode, TotalProfitForThisTraderType, NumberOfTradersOfThisType, AverageProfitPerTraderOfThisType
    # print(list(trader_types.keys()))
    for ttype in sorted(list(trader_types.keys())):
        n = trader_types[ttype]['n']
        s = trader_types[ttype]['balance_sum']
        dumpfile.write('%s, %d, %d, %f, ' % (ttype, s, n, s / float(n)))

    if lob['bids']['best'] is not None:
        dumpfile.write('%s, ' % (lob['bids']['best']))
    else:
        dumpfile.write('N, ')
    if lob['asks']['best'] is not None:
        dumpfile.write('%s, ' % (lob['asks']['best']))
    else:
        dumpfile.write('N, ')

    dumpfile.write('\n')


# create a bunch of traders from traders_spec
# returns tuple (n_buyers, n_sellers)
# optionally shuffles the pack of buyers and the pack of sellers
def populate_market(traders_spec, traders, shuffle, verbose):
    def trader_type(robottype, name):
        if robottype == 'GVWY':
            return Trader_Giveaway('GVWY', name, 0.00, 0)
        elif robottype == 'ZIC':
            return Trader_ZIC('ZIC', name, 0.00, 0)
        elif robottype == 'SHVR':
            return Trader_Shaver('SHVR', name, 0.00, 0)
        elif robottype == 'SNPR':
            return Trader_Sniper('SNPR', name, 0.00, 0)
        elif robottype == 'ZIP':
            return Trader_ZIP('ZIP', name, 0.00, 0)
        elif robottype == 'PRZI':
            return Trader_PRZI('PRZI', name, 0.00, 0)
        elif robottype == 'PRSH':
            return Trader_PRZI_SHC('PRSH', name, 0.00, 0)
        elif robottype == 'AA':
            return Trader_AA('AA', name, 0.00, 0)
        elif robottype == 'XGBOOST':
            return Trader_xgboost('XGBOOST', name, 0.00, 0)
        else:
            sys.exit('FATAL: don\'t know robot type %s\n' % robottype)

    def shuffle_traders(ttype_char, n, traders):
        for swap in range(n):
            t1 = (n - 1) - swap
            t2 = random.randint(0, t1)
            t1name = '%c%02d' % (ttype_char, t1)
            t2name = '%c%02d' % (ttype_char, t2)
            traders[t1name].tid = t2name
            traders[t2name].tid = t1name
            temp = traders[t1name]
            traders[t1name] = traders[t2name]
            traders[t2name] = temp

    n_buyers = 0
    for bs in traders_spec['buyers']:
        ttype = bs[0]
        for b in range(bs[1]):
            tname = 'B%02d' % n_buyers  # buyer i.d. string
            traders[tname] = trader_type(ttype, tname)
            n_buyers = n_buyers + 1

    if n_buyers < 1:
        sys.exit('FATAL: no buyers specified\n')

    if shuffle:
        shuffle_traders('B', n_buyers, traders)

    n_sellers = 0
    for ss in traders_spec['sellers']:
        ttype = ss[0]
        for s in range(ss[1]):
            tname = 'S%02d' % n_sellers  # buyer i.d. string
            traders[tname] = trader_type(ttype, tname)
            n_sellers = n_sellers + 1

    if n_sellers < 1:
        sys.exit('FATAL: no sellers specified\n')

    if shuffle:
        shuffle_traders('S', n_sellers, traders)

    if verbose:
        for t in range(n_buyers):
            bname = 'B%02d' % t
            print(traders[bname])
        for t in range(n_sellers):
            bname = 'S%02d' % t
            print(traders[bname])

    return {'n_buyers': n_buyers, 'n_sellers': n_sellers}


# customer_orders(): allocate orders to traders
# parameter "os" is order schedule
# os['timemode'] is either 'periodic', 'drip-fixed', 'drip-jitter', or 'drip-poisson'
# os['interval'] is number of seconds for a full cycle of replenishment
# drip-poisson sequences will be normalised to ensure time of last replenishment <= interval
# parameter "pending" is the list of future orders (if this is empty, generates a new one from os)
# revised "pending" is the returned value
#
# also returns a list of "cancellations": trader-ids for those traders who are now working a new order and hence
# need to kill quotes already on LOB from working previous order
#
#
# if a supply or demand schedule mode is "random" and more than one range is supplied in ranges[],
# then each time a price is generated one of the ranges is chosen equiprobably and
# the price is then generated uniform-randomly from that range
#
# if len(range)==2, interpreted as min and max values on the schedule, specifying linear supply/demand curve
# if len(range)==3, first two vals are min & max, third value should be a function that generates a dynamic price offset
#                   -- the offset value applies equally to the min & max, so gradient of linear sup/dem curve doesn't vary
# if len(range)==4, the third value is function that gives dynamic offset for schedule min,
#                   and fourth is a function giving dynamic offset for schedule max, so gradient of sup/dem linear curve can vary
#
# the interface on this is a bit of a mess... could do with refactoring


def customer_orders(time, last_update, traders, trader_stats, os, pending, verbose):
    def sysmin_check(price):
        if price < bse_sys_minprice:
            print('WARNING: price < bse_sys_min -- clipped')
            price = bse_sys_minprice
        return price

    def sysmax_check(price):
        if price > bse_sys_maxprice:
            print('WARNING: price > bse_sys_max -- clipped')
            price = bse_sys_maxprice
        return price

    def getorderprice(i, sched, n, mode, issuetime):
        if mode == 'random' and len(sched) > 1:
            # random and more than one schedule: choose one equiprobably
            s = random.randint(0, len(sched) - 1)
        else:
            # not random OR random but only 1 schedule: select the first
            s = 0

        # does the schedule range include optional dynamic offset function(s)?
        if len(sched[s]) > 2:
            offsetfn = sched[s][2]
            if callable(offsetfn):
                # same offset for min and max
                offset_min = offsetfn(issuetime)
                offset_max = offset_min
            else:
                sys.exit('FAIL: 3rd argument of sched in getorderprice() not callable')
            if len(sched[s]) > 3:
                # if second offset function is specfied, that applies only to the max value
                offsetfn = sched[s][3]
                if callable(offsetfn):
                    # this function applies to max
                    offset_max = offsetfn(issuetime)
                else:
                    sys.exit('FAIL: 4th argument of sched in getorderprice() not callable')
        else:
            offset_min = 0.0
            offset_max = 0.0

        pmin = sysmin_check(offset_min + min(sched[s][0], sched[s][1]))
        pmax = sysmax_check(offset_max + max(sched[s][0], sched[s][1]))
        prange = pmax - pmin
        stepsize = prange / (n - 1)
        halfstep = round(stepsize / 2.0)

        if mode == 'fixed':
            orderprice = pmin + int(i * stepsize)
        elif mode == 'jittered':
            orderprice = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
        elif mode == 'random':
            orderprice = random.randint(pmin, pmax)
        else:
            sys.exit('FAIL: Unknown mode in schedule')
        orderprice = sysmin_check(sysmax_check(orderprice))
        return orderprice

    def getissuetimes(n_traders, mode, interval, shuffle, fittointerval):
        interval = float(interval)
        if n_traders < 1:
            sys.exit('FAIL: n_traders < 1 in getissuetime()')
        elif n_traders == 1:
            tstep = interval
        else:
            tstep = interval / (n_traders - 1)
        arrtime = 0
        issuetimes = []
        for t in range(n_traders):
            if mode == 'periodic':
                arrtime = interval
            elif mode == 'drip-fixed':
                arrtime = t * tstep
            elif mode == 'drip-jitter':
                arrtime = t * tstep + tstep * random.random()
            elif mode == 'drip-poisson':
                # poisson requires a bit of extra work
                interarrivaltime = random.expovariate(n_traders / interval)
                arrtime += interarrivaltime
            else:
                sys.exit('FAIL: unknown time-mode in getissuetimes()')
            issuetimes.append(arrtime)

            # at this point, arrtime is the last arrival time
        if fittointerval and ((arrtime > interval) or (arrtime < interval)):
            # generated sum of interarrival times longer than the interval
            # squish them back so that last arrival falls at t=interval
            for t in range(n_traders):
                issuetimes[t] = interval * (issuetimes[t] / arrtime)
        # optionally randomly shuffle the times
        if shuffle:
            for t in range(n_traders):
                i = (n_traders - 1) - t
                j = random.randint(0, i)
                tmp = issuetimes[i]
                issuetimes[i] = issuetimes[j]
                issuetimes[j] = tmp
        return issuetimes

    def getschedmode(time, os):
        got_one = False
        for sched in os:
            if (sched['from'] <= time) and (time < sched['to']):
                # within the timezone for this schedule
                schedrange = sched['ranges']
                mode = sched['stepmode']
                got_one = True
                break  # jump out the loop -- so the first matching timezone has priority over any others
        if not got_one:
            sys.exit('Fail: time=%5.2f not within any timezone in os=%s' % (time, os))
        return (schedrange, mode)

    n_buyers = trader_stats['n_buyers']
    n_sellers = trader_stats['n_sellers']

    shuffle_times = True

    cancellations = []

    if len(pending) < 1:
        # list of pending (to-be-issued) customer orders is empty, so generate a new one
        new_pending = []

        # demand side (buyers)
        issuetimes = getissuetimes(n_buyers, os['timemode'], os['interval'], shuffle_times, True)

        ordertype = 'Bid'
        (sched, mode) = getschedmode(time, os['dem'])
        for t in range(n_buyers):
            issuetime = time + issuetimes[t]
            tname = 'B%02d' % t
            orderprice = getorderprice(t, sched, n_buyers, mode, issuetime)
            order = Order(tname, ordertype, orderprice, 1, issuetime, -3.14)
            new_pending.append(order)

        # supply side (sellers)
        issuetimes = getissuetimes(n_sellers, os['timemode'], os['interval'], shuffle_times, True)
        ordertype = 'Ask'
        (sched, mode) = getschedmode(time, os['sup'])
        for t in range(n_sellers):
            issuetime = time + issuetimes[t]
            tname = 'S%02d' % t
            orderprice = getorderprice(t, sched, n_sellers, mode, issuetime)
            order = Order(tname, ordertype, orderprice, 1, issuetime, -3.14)
            new_pending.append(order)
    else:
        # there are pending future orders: issue any whose timestamp is in the past
        new_pending = []
        for order in pending:
            if order.time < time:
                # this order should have been issued by now
                # issue it to the trader
                tname = order.tid
                response = traders[tname].add_order(order, verbose)
                if verbose:
                    print('Customer order: %s %s' % (response, order))
                if response == 'LOB_Cancel':
                    cancellations.append(tname)
                    if verbose:
                        print('Cancellations: %s' % cancellations)
                # and then don't add it to new_pending (i.e., delete it)
            else:
                # this order stays on the pending list
                new_pending.append(order)
    return [new_pending, cancellations]


def midprice(self, bid_p, bid_q, ask_p, ask_q):
    # returns midprice as mean of best bid and best ask if both best bid & best ask exist
    # if only one best price exists, returns that as mid
    # if neither best price exists, returns None
    mprice = None
    if bid_q > 0 and ask_q == None:
        mprice = bid_p
    elif ask_q > 0 and bid_q == None:
        mprice = ask_p
    elif bid_q > 0 and ask_q > 0:
        mprice = (bid_p + ask_p) / 2.0
    return mprice


def microprice(self, bid_p, bid_q, ask_p, ask_q):
    mprice = None
    if bid_q > 0 and ask_q > 0:
        tot_q = bid_q + ask_q
        mprice = ((bid_p * ask_q) + (ask_p * bid_q)) / tot_q
    return mprice


smithsAlpha = None
lastTrades = []
smithsAlphaMin = None
smithsAlphaMax = None
eqlbm = None
nLastTrades = 5  # N in AIJ08
ema_param = 2 / float(nLastTrades + 1)


def updateEq(price):
    # Updates the equilibrium price estimate using EMA
    global eqlbm
    if eqlbm == None:
        eqlbm = price
    else:
        eqlbm = ema_param * price + (1 - ema_param) * eqlbm
    return eqlbm


def updateSmithsAlpha(price):
    lastTrades.append(price)
    if not (len(lastTrades) <= nLastTrades): lastTrades.pop(0)
    smithsAlpha = math.sqrt(
        sum(((p - eqlbm) ** 2) for p in lastTrades) * (1 / float(len(lastTrades)))) / eqlbm
    return smithsAlpha


def market_session(sess_id, starttime, endtime, trader_spec, order_schedule, summaryfile, tapedumpfile, lobdumpfile,
                   dump_all, verbose):
    print(sess_id)
    orders_verbose = False
    lob_verbose = False
    process_verbose = False
    respond_verbose = False
    bookkeep_verbose = False
    populate_verbose = False

    # initialise the exchange
    exchange = Exchange()

    # create a bunch of traders
    traders = {}
    trader_stats = populate_market(trader_spec, traders, True, populate_verbose)

    # timestep set so that can process all traders in one second
    # NB minimum interarrival time of customer orders may be much less than this!!
    timestep = 1.0 / float(trader_stats['n_buyers'] + trader_stats['n_sellers'])

    duration = float(endtime - starttime)

    last_update = -1.0

    time = starttime

    pending_cust_orders = []

    if verbose:
        print('\n%s;  ' % sess_id)
    prev_time = 0
    while time < endtime:

        # how much time left, as a percentage?
        time_left = (endtime - time) / duration
        fname = 'train.csv'
        train_data = open(fname, 'a')

        # if verbose: print('\n\n%s; t=%08.2f (%4.1f/100) ' % (sess_id, time, time_left*100))
        trade = None

        [pending_cust_orders, kills] = customer_orders(time, last_update, traders, trader_stats,
                                                       order_schedule, pending_cust_orders, orders_verbose)

        # if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
        if len(kills) > 0:
            # if verbose : print('Kills: %s' % (kills))
            for kill in kills:
                # if verbose : print('lastquote=%s' % traders[kill].lastquote)
                if traders[kill].lastquote is not None:
                    # if verbose : print('Killing order %s' % (str(traders[kill].lastquote)))
                    exchange.del_order(time, traders[kill].lastquote, verbose)

        # get a limit-order quote (or None) from a randomly chosen trader
        tid = list(traders.keys())[random.randint(0, len(traders) - 1)]
        order = traders[tid].getorder(time, time_left, exchange.publish_lob(time, lob_verbose))

        # if verbose: print('Trader Quote: %s' % (order))
        output_flag = False
        if order is not None:
            if order.otype == 'Ask' and order.price < traders[tid].orders[0].price:
                sys.exit('Bad ask')
            if order.otype == 'Bid' and order.price > traders[tid].orders[0].price:
                sys.exit('Bad bid')
            # send order to exchange
            traders[tid].n_quotes = 1
            trade = exchange.process_order2(time, order, process_verbose)
            if trade is not None:
                # trade occurred,
                # so the counterparties update order lists and blotters

                traders[trade['party1']].bookkeep(trade, order, bookkeep_verbose, time)
                traders[trade['party2']].bookkeep(trade, order, bookkeep_verbose, time)

                # previous_transactions.append(trade['price'])

                # 2
                output_flag = True
                # 12
                output_eq = updateEq(trade['price'])
                # 13
                output_smithsAlpha = updateSmithsAlpha(trade['price'])
                # 14
                output_trade_price = trade['price']
                prev_time = time

                if dump_all:
                    trade_stats(sess_id, traders, summaryfile, time, exchange.publish_lob(time, lob_verbose))
                    # tapedumpfile.write('%s, %s, %s, %s, %s \n' % (trade['time'], trade['price'], trade['qty'], trade['party1'], trade['party2']))
            else:
                output_eq = 0
                output_smithsAlpha = 0
                output_trade_price = 0
            # traders respond to whatever happened
            lob = exchange.publish_lob(time, lob_verbose)

            # 1
            output_time = time
            # 3
            output_price = order.price
            # 4
            if lob['asks']['best'] and lob['bids']['best']:
                output_spread = lob['asks']['best'] - lob['bids']['best']
            else:
                output_spread = 0
            # 5
            output_midprice = lob['midprice']
            # 6
            output_microprice = lob['microprice']
            # 7
            if lob['bids']['best'] is None:
                output_best_bids = 0
            else:
                output_best_bids = lob['bids']['best']
            # 8
            if lob['asks']['best'] is None:
                output_best_asks = 0
            else:
                output_best_asks = lob['asks']['best']
            # 9
            if prev_time > 0:
                output_time_elapsed = time - prev_time
            else:
                output_time_elapsed = time
            # 10
            if output_midprice and output_microprice:
                output_imbalance = output_microprice - output_midprice
            else:
                output_imbalance = 0
            # 11
            output_quotes_quantity = lob['bids']['n'] + lob['bids']['n']

            # 14
            # output_trade_price = trade['price']

            lobdumpfile.write(' %s,"%s","%s", %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' % (
            lob['time'], lob['bids']['lob'], lob['asks']['lob'], output_price, output_flag, output_spread,
            output_midprice, output_microprice, output_best_bids, output_best_asks
            , output_time_elapsed, output_imbalance, output_quotes_quantity, output_eq, output_smithsAlpha,
            output_trade_price))
            if output_flag is True:
                train_data.write(' %s,"%s","%s", %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' % (
                lob['time'], lob['bids']['lob'], lob['asks']['lob'], output_price, output_flag, output_spread,
                output_midprice, output_microprice, output_best_bids, output_best_asks
                , output_time_elapsed, output_imbalance, output_quotes_quantity, output_eq, output_smithsAlpha))

            for t in traders:
                # NB respond just updates trader's internal variables
                # doesn't alter the LOB, so processing each trader in
                # sequence (rather than random/shuffle) isn't a problem
                traders[t].respond(time, lob, trade, respond_verbose)
        # prev_time = time
        time = time + timestep

    # session has ended

    if dump_all:

        # dump the tape (transactions only -- not dumping cancellations)
        exchange.tape_dump(sess_id + '_transactions.csv', tapedumpfile, 'w', 'keep', traders)

        # record the blotter for each trader
        blotterdumpfile = open(sess_id + '_blotters.csv', 'w')
        for t in traders:
            tid = traders[t].tid
            ttype = traders[t].ttype
            balance = traders[t].balance
            blot = traders[t].blotter
            blot_len = len(blot)
            csv = ''
            estr = "TODO "
            for b in traders[t].blotter:
                csv = csv + estr
            blotterdumpfile.write('%s, %s, %s, %s, %s, %s\n' % (sess_id, tid, ttype, balance, blot_len, csv))

        blotterdumpfile.close()

    # write trade_stats for this session (NB end-of-session summary only)
    trade_stats(sess_id, traders, summaryfile, time, exchange.publish_lob(time, lob_verbose))


#############################

# # Below here is where we set up and run a series of experiments


if __name__ == "__main__":

    # set up common parameters for all market sessions
    start_time = 0.0
    end_time = 1000.0
    duration = end_time - start_time


    # schedule_offsetfn returns time-dependent offset, to be added to schedule prices
    def schedule_offsetfn(t):

        pi2 = math.pi * 2
        c = math.pi * 3000
        wavelength = t / c
        gradient = 100 * t / (c / pi2)
        amplitude = 100 * t / (c / pi2)
        offset = gradient + amplitude * math.sin(wavelength * t)
        return int(round(offset, 0))


    # Here is an example of how to use the offset function
    #
    # range1 = (10, 190, schedule_offsetfn)
    # range2 = (200,300, schedule_offsetfn)

    # Here is an example of how to switch from range1 to range2 and then back to range1,
    # introducing two "market shocks"
    # -- here the timings of the shocks are at 1/3 and 2/3 into the duration of the session.
    #
    # supply_schedule = [ {'from':start_time, 'to':duration/3, 'ranges':[range1], 'stepmode':'fixed'},
    #                     {'from':duration/3, 'to':2*duration/3, 'ranges':[range2], 'stepmode':'fixed'},
    #                     {'from':2*duration/3, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
    #                   ]

    # The code below sets up symmetric supply and demand curves at prices from 50 to 150, P0=100

    range1 = (50, 150)
    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}
                       ]

    range2 = (50, 150)
    demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}
                       ]

    order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                   'interval': 30, 'timemode': 'drip-poisson'}
    # Use 'periodic' if you want the traders' assignments to all arrive simultaneously & periodically
    #               'interval': 30, 'timemode': 'periodic'}

    # buyers_spec = [('GVWY', 10), ('SHVR', 10), ('ZIC', 10), ('ZIP', 10), ('AA', 10)]
    # sellers_spec = [('GVWY', 10), ('SHVR', 10), ('ZIC', 10), ('ZIP', 10), ('AA', 10)]
    #buyers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('ZIP', 10), ('AA', 10), ('XGBOOST', 10)]
    #sellers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('ZIP', 10), ('AA', 10), ('XGBOOST', 10)]
    buyers_spec = [('AA', 20), ('XGBOOST', 20)]
    sellers_spec = [('AA',20), ('XGBOOST', 20)]

    traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

    # run a sequence of trials, one session per trial
    verbose = True

    # n_trials is how many trials (i.e. market sessions) to run in total
    n_trials = 6

    # n_recorded is how many trials (i.e. market sessions) to write full data-files for
    n_trials_recorded = 6
    fname = 'Mybalances3.csv'
    summary_data_file = open(fname, 'w')

    fname = 'Mytapes3.csv'
    tape_data_file = open(fname, 'w')
    tape_data_file.write('%s,%s,%s,%s,%s\n' % ('Time', 'Price', 'Qty', 'Party1', 'Party2'))

    fname = 'lob3.csv'
    # blotter_data_file = open(fname, 'w')
    lob_data_file = open(fname, 'w')
    lob_data_file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
    'Time', 'bids', 'asks', 'order price', 'flag', 'spread', 'midprice', 'microprice'
    , 'best_bids', 'best_asks', 'time_elapsed', 'imbalance', 'quotes_quantity', 'equilibrium price', 'smithsAlpha',
    'trade price'))

    trial = 1

    while trial < (n_trials + 1):
        trial_id = 'sess%04d' % trial

        if trial > n_trials_recorded:
            dump_all = False
        else:
            dump_all = True

        market_session(trial_id, start_time, end_time, traders_spec, order_sched, summary_data_file, tape_data_file,
                       lob_data_file, dump_all, verbose)
        # tdump.flush()
        trial = trial + 1
    summary_data_file.close()
    tape_data_file.close()
    lob_data_file.close()
    # tdump.close()

'''
if self.job == 'Bid':
    quoteprice = loaded_model.predict(loaded_data)[0]
    print(quoteprice)
    while (quoteprice < output_price):
        print(quoteprice)
        print(output_price)
        print("---")
        quoteprice = loaded_model.predict(loaded_data)[0]
else:
    quoteprice = loaded_model.predict(loaded_data)[0]
    print(quoteprice)
    print(output_price)
    while (quoteprice > output_price):
        quoteprice = loaded_model.predict(loaded_data)[0]
order = Order(self.tid, self.job, quoteprice, self.orders[0].qty, time, lob['QID'])
'''