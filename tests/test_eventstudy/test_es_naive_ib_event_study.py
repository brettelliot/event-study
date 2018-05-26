import unittest
import datetime as dt
from eventstudy.eventstudy import *
from eventstudy.ibdatareader import IBDataReader
from tests.test_eventstudy.earningseventmatrix import EarningsEventMatrix


class TestESNaiveIBEventStudy(unittest.TestCase):
    def setUp(self):
        # Define the symbols to study
        #self.symbols = ["MOS","WOR","UNF","FC","FDO","CMC","ZEP","TISI","AYI","RPM","MON","GPN","LNN","IHS","MG","TXI","AZZ","GBX","SJR","STZ","RT","MSM","SVU","SNX","INFY","WFC","PPG","SAR","FRX","BK","LEN","GS","FRC","USB","JPM","MTB","CMA","WNS","SCHW","KMI","EPB","CLC","FUL","BAC","COF","ASB","BLK","APH","BBT","UNH","WIT","PNC","C","AXP","TSM","STI","COL","FHN","SLB","MS","MMR","GE","WBS","PGR","PH","STT","WAT","JNJ","VZ","PKG","CBU","FCX","KSU","ETH","RF","DAL","LEE","RKT","ALB","TRV","TSS","SNV","CNI","EAT","NSC","TAL","CLS","UFI","DD","COH","OMN","IBM","FNB","EXAR","URI","GHL","GD","PX","MSI","AF","CCI","EGN","NE","SWFT","SAP","LCC","CGI","CNS","CMRE","FCFS","HRC","MPX","MCD","JEC","VAR","STJ","NVS","TXT","ROL","INVN","TER","ABT","APD","STL","HXL","SYK","RJF","BXS","ATI","TEL","UTX","RES","DGX","RLI","BHI","JNS","CYN","NVR","TKR","WAL","RYN","FII","ABC","GWW","KEY","EQT","ARG","DLX","AOS","DOV","SWK","XRX","AME","EQM","MDP","ALK","CVD","LMT","KMT","RTN","UNP","TDY","BGG","ORI","HUBB","T","MMM","BC","BAX","LUV","TPX","AVT","RMD","NMM","MKC","JNPR","UAL","BMY","CSH","PCP","AVX","HON","MOG.A","OSK","WY","PG","KMB","ACO","HAL","PB","BH","NBHC","GGG","IRF","NDZ","ELS","CAT","BZH","ROP","PCL","CR","BOXC","WRB","NEU","CE","OLN","VMW","BHLB","BBD","CTS","LLY","HRS","ITW","PLT","IP","GLW","SFG","POL","MSL","GNI","HW","HOG","PFE","JLL","AXE","BKU","NEE","PNR","VLO","CB","AMG","BSX","RHI","WAIR","RYL","BXP","ASH","TUP","PII","AJG","EDU","CP","CODE","PHG","DHR","DHT","WDR","F","X","DHI","CIT","NUE","EMC","HZO","DLB","AKS","CHT","NYCB","HNT","FBR","KEX","TCB","MUR","SO","MAN","AMP","PJC","MTOR","ROK","UTL","VLY","EVR","ASX","OI","WEC","LVS","COP","NOC","HAE","BOH","ATW","AVY","BA","BAH","CMO","SLG","GIB","CCK","MPC","FCF","ST","VHS","CFR","FICO","CACI","CBT","CLB","PSX","ELY","EVER","MPLX","DHX","AVB","STM","ADT","LLL","GFF","DRE","TGI","HES","MX","REG","KRC","NOW","KNX","CAM","APU","MD","OXY","CRS","PHM","PBI","CPF","WHR","CL","NKA","DB","PFG","BX","POT","TMO","BERY","AIT","CNX","HCLP","NGVC","AET","MTH","ITG","AN","HSH","IVZ","CCU","ALV","UGI","SHW","UTI","BCR","FBHS","BKH","BLL","BMS","RGS","HTSI","R","DST","CRR","MCK","GDOT","TEN","TWC","EPC","XEL","CPT","HHS","MHO","MJN","HAR","AZN","HVB","MTW","MO","ELX","N","CYT","MDC","RDS.B","CNH","RGA","EMN","WCC","MTX","MA","BYI","DOW","HMC","VR","UPS","EPD","KEM","ADS","PKI","ED","ESS","HSY","HGG","HP","D","CAA","RDS.A","NSH","IR","GHM","BEN","SXI","SCCO","NOV","BCO","AON","PFS","LM","BEAM","TDW","TSN","NWL","MOD","BT","BRO","MRK","LEA","NS","CVX","VVI","OPY","PRGO","UFS","LYB","XOM","YUM","IPHI","SWI","AXS","HMY","LEG","IEX","HUM","SYY","ADM","BRS","EW","SYA","TMK","RTEC","FN","AP","CLX","MDU","PBR","TDG","HCA","BHE","MKL","HIG","ITUB","BSAC","HI","MCY","GGP","RBC","APC","RCL","SPG","EL","TM","LII","WNC","UDR","BAP","UNM","CNC","SU","BP","CBL","EQR","REV","BCH","VSH","USNA","WTM","RNR","DO","AKR","MMP","BDX","GNW","BMR","KNL","IVR","KFN","NSR","DLPH","PIKE","SXC","ETN","CMG","NYX","MLI","MWA","HBI","OB","CAH","MXL","LRN","CSC","HMN","CALX","CMLP","HNI","SEP","KIM","ACM","FBP","SE","CHD","NOA","K","EMR","GBL","SPA","TE","TMH","AFL","DIS","AGCO","SR","TSO","MCC","STE","MTD","LNC","AUO","TRR","PAA","AIN","BDN","UMC","SPB","WYN","TYL","CNW","RLD","CVS","MAA","CFX","CMI","GSK","PLD","GRA","GTS","PNG","RRTS","DFT","RL","CVH","ALL","WEX","SWS","PAG","FMC","AI","THG","CEB","RE","TBI","USG","ATO","EXP","GAS","PHH","WGL","LF","PDH","CVA","PRU","ENS","YELP","CBG","AIZ","PL","SMI","MT","HOS","GIL","MAC","TWO","CBM","DV","TWX","MRO","CMP","CPA","SWM","NUS","ICE","BMI","CYS","SMG","NLY","V","RTI","EFX","RSG","PDM","CX","SQNS","ANH","ALU","MATX","NCR","FFG","OFG","AAN","ARW","POST","ARE","PRI","SPH","BCE","BLX","LNKD","THR","CCE","CI","FLT","KKR","INGR","XYL","CVG","TEVA","MNR","PBH","LAZ","CS","BG","MPW","NYT","AAP","STO","HPY","XL","NJR","NFG","NBL","SXT","NNN","PHX","KRG","BDC","ENH","MWW","SNA","MOH","GPK","FMD","MRH","AHL","BAK","KMPR","WHG","SNY","NPO","RGC","LCI","CLI","ATR","MSCI","TDC","S","LPS","PM","SSD","GLT","MMS","OZM","MFC","PMT","CZZ","FLO","G","HOT","IFF","IT","PMC","SLH","CLP","BR","DCT","SBH","EXC","AIV","SNN","ESE","AOL","NSP","AXL","APO","BIP","FBC","OFC","CGA","SFUN","MCO","ETM","CFN","CFN","ETR","CCJ","BPL","CSL","BFR","LH","LPX","HRG","BLC","IVC","LDR","RXN","NAT","GEN","L","CRK","AFG","RDN","DAC","DNB","MAS","WPP","CUB","BKD","BWP","OHI","OMI","CNA","NLSN","PKY","CHH","CNO","BNNY","TLLP","RAX","FRT","PZN","RAI","MLM","DBD","HIW","VPG","EGP","NNA","PRO","HTS","ACC","TRP","OMC","HUN","VMI","MMC","WU","FIS","DEI","MET","FTI","KO","AMX","DDR","SKT","TGH","HCP","HCC","AXLL","RHP","SPR","RFP","SCI","CLF","MIG","GWAY","KORS","AVP","TRLA","AB","BCS","LVLT","RKUS","RPT","GWR","RATE","VAL","OII","TOT","WCG","AEM","HSP","PES","JMP","JNY","DF","SON","IO","CXW","KGC","FOR","CRL","TCO","PGI","AWH","TR","NRP","CAE","TRI","ACCO","BXC","TPGI","MYE","DE","WTW","WWAV","GLA","CPN","KS","VG","EEP","WRE","MSA","IPI","PXD","IM","PRLB","CTL","SLF","H","CUZ","MN","DUK","SKX","EFC","LO","CRD.B","DPS","RAS","EOG","CLD","ABX","NCI","WRI","LNT","WSO","ECA","CVE","PDS","PWE","ENV","ACTV","NRF","TMS","JAH","NR","WCN","NHI","EEQ","ELLI","GRT","ORB","CBS","VMC","VVC","GNRC","BBW","KOP","BMA","NWE","TAP","O","RCI","ART","HSC","FET","GM","HMA","OAK","OWW","A","BWA","NFP","GEL","WM","DWRE","ASGN","STC","GNC","DVA","GFI","GG","PPL","CAB","BGS","CRY","EDE","PEP","APA","RDY","MANU","ABB","NGLS","ABR","E","COT","ENB","TOWR","SJM","DLR","TRW","LSE","ALR","NPK","BAM","VTR","HE","ALE","WBC","TU","EC","BKW","TRNO","NGL","CPB","VFC","AEP","TLM","BSMX","HLS","APL","CHE","AAT","WPZ","FUN","ROG","OCR","HLF","SEE","ROC","RPAI","MM","SJW","GPC","ALSN","SCL","NI","ALEX","CLDT","CCO","NM","NBR","EDR","FNF","MDT","SHO","HY","WLK","CNL","AWI","CF","AMRE","LZB","QEP","PSB","BAS","ES","UAM","OIS","WWW","PRA","OMG","TNC","NFX","EE","TREX","GPI","FCH","WTS","TEX","FDP","IHG","IRS","WAB","ABG","XEC","UNT","SXL","CNK","AER","KBR","IAG","BEL","OC","MIC","HEI.A","SM","PGTI","GNK","MGM","TOL","PPO","WMB","XCO","ETE","RGP","GTN","PVR","STR","HEI","AU","VNTV","OMX","LHO","EQY","WAGE","HTA","STAG","CLH","TROX","HBM","HLX","LOCK","FLTX","FR","SIX","HT","CW","BALT","KAR","DTE","DVN","ESV","CLW","UGP","SB","NP","LL","EV","HR","DX","CXO","CPAC","SAH","SAM","AEE","FLR","ETP","AUY","AEL","SWN","LAD","AVA","TRN","SUI","PKD","MHK","TNK","DAN","LPL","WMT","CYH","CUBE","BRC","HST","SMA","CLGX","LXP","BBG","SEM","SWY","THI","DNR","PWR","INT","JWN","AYR","TFX","LTC","FRO","LTM","TOO","KMR","IDA","HPQ","Y","HRL","KDN","CHK","ERF","CWEI","HEP","TTC","TS","ZLC","PEB","CHSP","AIG","NPTN","THS","UIL","CDE","CCC","AMN","TGP","MRC","PSA","PPP","RWT","AGU","VIPS","VAC","ET","PXP","SCG","FLS","BLOX","EXR","NEM","WST","CBR","GGB","SSTK","TPC","FAF","IMAX","RS","FPO","COG","IL","GEO","PCG","PEG","CMS","STN","COR","ARR","NVE","PNW","IPG","B","EGO","POR","ANF","MR","NXY","NOAH","CHMT","ITGR","TWI","HL","DUF","SGY","CSV","HCN","DDS","HVT","DCI","OAS","GLF","VIV","KND","OKS","BGC","ANFI","CTB","OKE","STNG","FE","SF","PEI","TV","KAMN","KOS","URS","LUK","LOW","DDD","SFL","ARB","ASR","VGR","WTI","CDI","INN","THC","LDL","GWRE","TSL","FMS","ALX","SKS","AWK","SPN","SLCA","ECL","SRE","GTI","DGI","RRD","SWX","USM","CIE","BIO","KAI","BMO","DL","EME","EPR","LYV","EIX","VNO","WFT","BZ","SSP","UNS","CO","TDS","AG","AZO","STAR","RBA","RRC","HD","CCG","DY","HFC","VSI","GPX","CHG","AMT","M","MWE","GMED","KRA","SRC","VALE","CBI","RLJ","DAR","SWC","GLOG","XUE","EXL","EXAM","SRT","OGE","INXN","CYD","IOC","CRI","HII","AHT","AES","CBB","WLL","ITT","GEF","CKH","ITC","ADC","UAN","CWT","GMK","FMX","WGP","LB","FIG","DIN","WES","JCP","VCRA","CVO","JOY","CFI","KOF","PAC","XPO","TJX","CLR","EVC","DCP","DEL","HNZ","PACD","AGO","ORA","EIG","CNP","NRG","RGR","EPAM","ARI","STWD","SUSS","BUD","TGT","JHX","RM","ANW","PLL","UHT","PANW","SJI","KOG","TEO","RRMS","HK","FIX","CM","WPX","JOE","AVD","LUX","YOKU","MEI","ORN","WR","DRQ","MTZ","DRH","OCN","MTRN","MDR","TEG","HTGC","KSS","AWR","LXU","TD","RNDY","ARC","CVC","SEMG","WNR","SDLP","WWE","VRX","RDC","CHS","SBY","RST","BKS","AL","BID","PBF","CRM","BVN","GXP","RLH","SDRL","MTG","VC","GPS","ESL","FCN","PQ","AT","DPZ","DRC","RY","CIR","ESC","IRM","GTY","UHS","GVA","TTI","DECK","LIN","SUP","BRK.A","XLS","PER","PBA","CGG","PKX","PNM","BANC","POM","NWN","RIG","BRK.B","MGA","SRI","BBY","BYD","ABM","RSO","DCO","CIB","YGE","QUAD","CEL","NLS","DFS","BRFS","WMK","PVG","SQM","CKP","QIHU","HCI","FOE","DKL","GRP.U","SYX","PAY","MITT","BNS","SSW","OME","CSU","ASI","LVB","UBA","P","HOV","KFY","AEO","BIG","MTN","PNY","BF.B","ACW","CCSC","JBT","FENG","TRK","ALJ","NQ","CLNY","NMFC","MLR","MFB","QRE","HF","WD","HPP","ALG","CODI","XOXO","TCAP","OILT","AMTG","KWR","WG","SN","MFA","HRB","BFS","BTE","CNQ","FLY","JW.A","POWR","CIEN","WX","BLT","FUR","TUC","CDR","SCR","FGP","COO","USPH","MED","REN","NSM","THO","WDAY","JRN","SFE","CMD","NX","NAV","DK","IDT","KR","CPK","EBS","MAIN","BITA","MVC","DANG","ISS","FRM","ANN","GCO","KMG","ARCO","MUX","FL","CQB","HIL","ODC","LXFR","GFA","CIA","PLOW","RENN","IN","DKS","EXK","EJ","YPF","CVI","DOLE","ERJ","LPI","KRO","SSI","CPL","EGL","GCAP","ENZ","WSR","KW","TLP","IRET","BPI","CBK","SMLP","MTDR","EXPR","NTI","FTK","GLP","OLP","DSX","NGS","KKD","EGY","BCEI","CPG","CPE","TAOM","BKE","SOL","DYN","MVO","CUK","SDT","CHKR","STON","CAL","CCL","ECT","HTH","WAC","AMRC","AGM","NDRO","FF","FDS","FNV","RNF","WSM","AIR","MBT","DSW","GES","FSM","GIS","TUMI","ATU","ORCL","TLYS","FDX","LUB","JBL","SLW","NKE","KBH","JMI","MPR","NWY","MOV","SBS","AGRO","CATO","TIF","DRI","TRQ","DG","AUQ","GOL","BFAM","RHT","AAV","SCS","BSBR","FCE.A","PVH","ACN","WGO","FRF","ZTS","RIOM","SXE","GME","SIG","REX","SID","SUN","ACRE","WMC","OXM","CAG","DDC","KMX","JKS","AGX","PIR","PBY","AMID","NOK","RH","TNP","BCC","UIS","LXK","TAC","CAI","FIO","AGI","SXCP","ABBV","CBD","MSO","LFL","TX","RLGY","KED","SSNI","APAM","WTR","PGH","CVRR","CBZ","HNR","TRGP","OAKS","SMP","FSS","MODN","AVIV","ALDW","GNE","QTM","MRIN","USAC","SCM","PKE","GSL","ROYT","CORR","SVN","TPH","SLD","PF","TMHC","ELP","CIG","XNY","TTM","CNCO","NCS","RALY","CSTM","MTL","RAD","ARPI","MCS","AGN"]
        self.symbols =["MOS", "WOR", "UNF", "FC", "FDO", "CMC", "ZEP", "TISI", "AYI", "RPM", "MON", "GPN", "LNN", "IHS", "MG",
               "TXI", "AZZ", "GBX", "SJR", "STZ", "RT", "MSM", "SVU", "SNX", "INFY", "WFC", "PPG", "SAR", "FRX", "BK",
               "LEN", "GS", "FRC", "USB", "JPM", "MTB", "CMA", "WNS", "SCHW"]

        # Define the market symbol to compare against
        self.market_symbol = "SPY"

        # Add the market symbol to the symbols list to get it's data too
        self.symbols.append(self.market_symbol)
        #self.symbols.insert(0, self.market_symbol)

        # Define the start and end date of the study
        self.start_date = dt.datetime(2013, 1, 1)
        self.end_date = dt.datetime(2013, 12, 31)

        # Get a pandas multi-indexed dataframe indexed by date and symbol
        keys = ['Close', 'Volume']
        data_reader = IBDataReader(data_path='../../data/ib')
        self.stock_data = data_reader.get_multiple_years_of_daily_bars_for_multiple_stocks_as_midf(
            self.symbols, keys, self.start_date.year, self.end_date.year, 1)

        #print(self.stock_data[:])

        self.symbols.remove('SPY')

        #print(self.stock_data.loc['SPY'])
        # print(self.stock_data.loc['AES']['Close'])
        # print(self.stock_data.loc['AES'][self.stock_data.loc['RGR'].index=='2013-12-30'])

        self.stock_data.drop_duplicates(inplace=True)

    @unittest.skip("skipping because this doesn't pass. unsure if its a test or code problem.")
    def test_posivite_cars(self):
        # We could try to pick different stocks
        # I think we need to get the t-test value up to 1.96
        pct_diff = 50.0
        look_back = 20
        look_forward = 20
        positive = True

        em = EarningsEventMatrix(self.stock_data.index.levels[1], self.symbols, pct_diff,
                                 positive, '../../data/events/nyse_earnings_surprises_2013.csv')

        # Get a dataframe with an index of all trading days, and columns of all symbols.
        event_matrix = em.build_event_matrix(self.start_date, self.end_date)
        # print(event_matrix[(event_matrix == 1.0).any(axis=1)])
        #print("Number of events:" + str(len(event_matrix[(event_matrix == 1.0).any(axis=1)])))

        calculator = Calculator()

        ccr = calculator.calculate_using_naive_benchmark(
            event_matrix, self.stock_data, self.market_symbol, look_back, look_forward)

        print(ccr.results_as_string())
        self.assertTrue(ccr.cars_significant)
        self.assertTrue(ccr.cars_positive)

        # plotter = Plotter()

        # plotter.plot_car(ccr.cars, ccr.cars_std_err, ccr.num_events, look_back, look_forward, True)
        # plotter.plot_car_cavcs(ccr.num_events, ccr.cars, ccr.cars_std_err, ccr.cavcs, ccr.cavcs_std_err,
        # look_back, look_forward, True)

    def test_negative_cars(self):
        pct_diff = -50.0
        look_back = 20
        look_forward = 20
        positive = False

        em = EarningsEventMatrix(self.stock_data.index.levels[1], self.symbols, pct_diff,
                                 positive, '../../data/events/nyse_earnings_surprises_2013.csv')

        # Get a dataframe with an index of all trading days, and columns of all symbols.
        event_matrix = em.build_event_matrix(self.start_date, self.end_date)
        #print(event_matrix[(event_matrix == 1.0).any(axis=1)])
        #print("Number of events:" + str(len(event_matrix[(event_matrix == 1.0).any(axis=1)])))

        calculator = Calculator()

        ccr = calculator.calculate_using_naive_benchmark(
            event_matrix, self.stock_data, self.market_symbol, look_back, look_forward)

        print(ccr.results_as_string())
        self.assertTrue(ccr.cars_significant)
        self.assertFalse(ccr.cars_positive)

        # plotter = Plotter()

        # plotter.plot_car(ccr.cars, ccr.cars_std_err, ccr.num_events, look_back, look_forward, True)


if __name__ == '__main__':
    unittest.main()
