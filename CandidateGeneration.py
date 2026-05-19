import re
import numpy
from sentence_transformers import SentenceTransformer, SimilarityFunction

class TipSheet:
    def __init__(self, title: str, tags: list, summary: str):
        self.title = title
        self.tags = set(tags)
        self.summary = summary
    

LOW_VALUE_PHRASES = [
    r"\bcalifornia\b",
    r"\bassembly\b",
    r"\bbill\b",
    r"\bcommittee\b",
    r"\bpasses\b",
    r"\badvances\b",
    r"\bapproves\b",
    r"\bappropriations\b",
    r"\bunanimously\b"
]

def clean_text(text: str) -> str:
    text = text.lower()

    # Remove repetitive low-information phrases
    for phrase in LOW_VALUE_PHRASES:
        text = re.sub(phrase, "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def jaccard_similarity(a, b):
    if not a and not b:
        return 0.0

    return len(a & b) / len(a | b)


# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.EUCLIDEAN)

tipsheets = [
        TipSheet(
            "California Assembly Judiciary Committee Approves Bill Protecting Homeowners' Right to Cooling Systems",
            ["Housing", "Budget", "Enviornment", "Judiciary", "Utilities"],
            "AB 1684 makes it unlawful for homeowners associations (HOAs) and other common‑interest development rules or deed restrictions to prohibit or unreasonably limit a homeowner from installing, upgrading, replacing, or using a cooling system in their own unit or lot. \"Cooling system\" can include window or portable air conditioners, evaporative (swamp) coolers, fans, heat pumps, and similar technologies, so long as they meet applicable state and local building, health, and safety rules. The bill also stops associations from charging fees, forcing homeowners to use a specific product or contractor, or claiming rebates or commissions related to a homeowner’s cooling system. Exceptions apply if the installation would violate law or if a required permit is denied. Associations may still require homeowners to repair any damage to common areas or other homes caused by the cooling system. If an association willfully violates the law, the homeowner can recover actual damages, a civil penalty (up to $2,000), and reasonable attorneys’ fees and court costs."
            ),
        TipSheet(
            "California Assembly Approves Santa Fe Springs Municipal Water Utility Sale Bill",
            ["Los Angeles", "Housing", "Finance", "Taxations", "Judiciary", "Economy", "Utilities", "Water"],
            "This bill lets the City of Santa Fe Springs, until January 1, 2032, sell its municipal water utility to another public water system to consolidate service, but only under strict conditions designed to protect customers and the public. The sale would require a four‑fifths vote of the City Council and cannot be for less than fair market value. It can only proceed if the city’s water supply is contaminated or otherwise unsafe, or the city lacks the technical, managerial, or financial ability to treat or replace the supply, or if continuing to operate would create an unreasonable cost for ratepayers — with those facts supported by an independent financial analysis (per Proposition 218). The acquiring system must border the current service area, the consolidation must be economically feasible for customers of the selling system, service must continue without interruption or loss of quality, and customers must be told the first‑year post‑consolidation rate (with any future increases phased in). The city must give public notice, allow 45 days for written and oral protests, and consider them; if at least 10% of “interested persons” (residents or nonresident ratepayers) protest, the city must hold an election and the sale needs a majority vote of those voting, and if 50% or more protest the sale is halted for one year. The bill also states the Legislature’s finding that this special rule is needed for Santa Fe Springs. "
            ),
        TipSheet(
                "California Senate Committee Advances Bill Requiring Water Suppliers' Wildfire Response Plans",
                ["Public Safety", "Utilities", "Water"],
                "SB 1153 requires urban retail water suppliers that serve areas designated as high or very high fire hazard zones to add wildfire-specific procedures to their disaster and emergency response plans starting January 1, 2028. The plans must identify mitigation measures (equipment, actions, and policies to reduce wildfire impacts on water supply), steps to prepare (identify critical infrastructure, coordinate with local emergency responders, and assess resilience and backup power options), actions to respond (including tank preparation during red-flag warnings and customer communications), and recovery steps (damage assessments and long-term adaptation). Suppliers must share these plans with the county Office of Emergency Services, while certain sensitive infrastructure information can be kept confidential. The bill also clarifies that water systems are not designed as wildfire suppression systems, that loss of water supply or pressure during a wildfire is not by itself a primary legal cause of wildfire damages, and it does not impose a duty to design systems for firefighting; negligent operation remains actionable. (The measure includes legislative findings about protecting sensitive operational information and may make violations a misdemeanor for covered suppliers.)"
            ),   
        TipSheet(
            "California Assembly Unanimously Passes Water District Board Compensation Bill",
            ["Housing", "Finance", "Taxations", "Judiciary", "Utilities", "Water"],
            "This bill temporarily lets large water districts (those serving at least 90,000 people) pay their board members for up to 15 days in a calendar month instead of the current 10-day limit. It keeps the existing rule that any pay increases above $100 per day can be limited to 5% per year, and it requires boards that pay for more than 10 days a month to adopt an annual, written policy—backed by substantial evidence—explaining why the extra compensated days are necessary for the district to operate effectively. The change is temporary and expires on January 1, 2032."
            ),
        TipSheet(
            "California Assembly Committee Passes Bill Easing Environmental Review for Housing Projects",
            ["Enviornment", "Water", "Wildlife", "Economy", "Education", "Housing"],
            "This bill changes California’s environmental review law (CEQA) to make it easier and faster to approve certain housing projects. Under the bill, many locally approved actions for qualifying affordable housing — and now housing built for students, faculty, or staff at the University of California, California State University, or California Community Colleges — would be exempt from full CEQA review if they meet specific requirements for affordability, labor standards, location (near transit and local services), safety, and basic environmental checks. Instead of the prior rule that relied on a specific finding about tribal cultural resources on vacant sites, the bill requires the lead agency to notify and consult with California Native American tribes before approving these projects and allows the agency to add conditions to avoid or reduce impacts to tribal cultural resources. It also shifts some confirmation duties to the lead agency, extends the temporary exemption through January 1, 2037, and requires filing a notice of exemption; the bill notes the new duties are a local responsibility and states no separate reimbursement is required."
            ),
        
        TipSheet(
            "Committee Vote California Appropriations Committee Approves Bill to Enhance Affordable Housing Density Bonus",
            ["Housing", "Budget", "Enviornment"],
            "This bill updates California’s Density Bonus Law to make it easier and faster for developers who include affordable or senior housing to get extra homes and other regulatory relief. If a housing project meets the bill’s affordability and other criteria, the city or county must grant the density bonus, requested incentives/concessions, and waivers or reductions of development standards for the sites that are part of that same project, and must adopt procedures and tell the applicant—when the application is deemed complete—what bonus, parking ratio, and incentives it is eligible for. The bill revises how incentives and density increases are calculated, clarifies parking reductions for qualifying projects, and broadens the definition of \"moderate‑income\" to include other lower‑income categories. Importantly, qualifying projects that meet certain CEQA exemption rules become \"use by right\" and are subject to ministerial (non‑discretionary) review—so local governments cannot require extra studies, discretionary approvals, or environmental review under CEQA for those aspects—and these rules apply statewide, including charter cities."    
            ),
        TipSheet(
            "California Appropriations Committee Reviews Bill Targeting 30% Reduction in Electricity Costs",
            ["Taxations", "Law", "Finance", "Economy", "Utilities", "Public Safety"],
            "This bill directs the California Public Utilities Commission to produce a report, due by January 1, 2028, with recommendations to reduce the price customers pay for electricity by at least 30% per kilowatt-hour. To do that, the commission must review utility \"public purpose\" programs and eliminate or reform ones that are not cost-effective, consider changes to the climate credit suggested in its response to Executive Order N‑5‑24, audit utility wildfire‑mitigation costs and recommend removing any unreasonable costs from rates, and evaluate specific programs listed in that response for cost-effectiveness. The report must also assess any remaining restitution shortfalls owed to victims of utility‑caused wildfires before July 12, 2019 and recommend ways utilities can address those shortfalls — but it may not propose recovering restitution payments by raising customer rates."
            ),

        TipSheet(
            "California Committee Approves Bill for Ontario Sports Empire Development",
            ["Housing", "Finance", "Taxations", "Judiciary", "Economy", "Entertainment", "Sports"],
            "This bill creates a narrow, one-off exception to California’s Surplus Land Act for specific parcels owned by the City of Ontario so the city can more easily develop and operate a large, master‑planned sports and entertainment district (the \"Ontario Sports Empire\" project) with uses like stadiums, youth and community sports fields, public plazas, parking and transit infrastructure, hotels and visitor-serving businesses, and supporting operations. To qualify, the land must be identified in the city’s adopted planning documents, the city council must make public findings that the disposition is consistent with the master plan and furthers public benefits, the parcels are limited in total size (up to 199 acres) and were not acquired by eminent domain, and any disposition must include a recorded covenant restricting future use. The exemption does not apply to standalone residential developments; if 10 or more homes are built there, at least 15% must be affordable to lower‑income households and subject to long‑term affordability rules. The city must notify the state housing department 30 days before disposing of the land, deposit a portion of the land’s value into a local housing set‑aside account (or meet specified prior-housing performance tests), and spend those funds on affordable housing within a short timeframe or have the funds revert to the state. The bill aims to balance allowing coordinated, phased development of a major regional public project with protections to support affordable housing and enforceability if the city fails to follow the rules."    
            ),
        TipSheet(
            "California Assembly Committee Approves Bill Tightening Election Security Regulations",
            ["Elections", "Law", "Judiciary", "Rules", "Public Safety"],
            "This bill is designed to protect the administration and security of elections by limiting when and how law enforcement and armed or security personnel can interact with voting places, ballots, voter lists, and voting equipment. It requires local elections officials to immediately notify the Secretary of State and Attorney General if a court order is being executed to search or seize voting machines, voter lists, ballots, or other election materials. The bill bars peace officers from interfering with how elections are run (for example, trying to decide who is or is not qualified to vote or imposing rules that conflict with state law), prevents law enforcement agents from serving as mail‑ballot observers or from accessing voter rosters or certified voting technology unless authorized by a court or for certain narrowly defined fraud investigations (while still allowing security or logistical support under written agreements), and strengthens prohibitions and penalties against removing voted ballots from election custody or stationing armed or unauthorized security or military personnel at polling places. It also gives the Attorney General, Secretary of State, and local elections officials tools to enforce these protections and requires the Attorney General to publish guidance on how to respond to law enforcement requests around ballot areas. The bill takes effect immediately."
            ),
        TipSheet(
            "Assembly Member Hoover Breaks GOP Ranks on Job Order Contracting Bill Approval",
            ["Education", "Finance", "Budget", "Economy", "Employment"],
            "AB 1809 makes permanent the statewide rules that let school districts and community college districts use \"job order contracting\" (a master contract with a contractor under which the district issues individual task orders for specific repair, modernization, or construction tasks) by removing a 2027 sunset date. It keeps conditions on that authority: districts must have project labor agreements that cover larger public works, contractors on job orders over $25,000 must commit to using a skilled and trained workforce (unless a project labor agreement already requires that), and neither a school nor a community college district may use job order contracting if it would increase the total cost of the project. The bill also requires community college districts to prepare execution plans and select projects that will save money, limits any master task‑order or job‑order contract to five years (while allowing individual task orders issued under them to remain valid after the master contract expires), and preserves contractor disclosure requirements."
            ),
        TipSheet(
            "California Assembly Committee Approves Boating Safety Bill Strengthening Regulations",
            ["Taxations", "Law", "Finance", "Public Safety", "Transportation", "Water"],
            "This bill makes several changes to strengthen boating safety and enforcement. It expands the definition of a \"for‑hire\" vessel so any motorboat carrying one or more paying passengers (instead of only those carrying more than three) is treated as a for‑hire vessel; it requires, as federal law allows, that all vessels using California waters be registered; it adds that boat operators must display a ski flag to warn of a swimmer in the vicinity (a violation remains an infraction with a fine up to $15); it bars courts from delaying, staying, or dismissing criminal proceedings in cases of operating a vessel under the influence by allowing the defendant to attend education or treatment programs (matching existing rules for vehicle DUI); and it authorizes trained peace officers to issue a written “notice to appear” (a citation) when an investigation shows a boating law violation was a factor in a vessel accident. The bill notes these changes expand the scope of crimes/infractions and makes no state reimbursement for related local costs."
            ),
        TipSheet(
            "California Assembly Appropriations Committee Approves Streamlined Building Permit Bill",
            ["Housing", "Finance", "Taxations", "Judiciary", "Business", "Economy", "Public Safety"],
            "AB 2418 is intended to speed up review and inspections for many nonresidential building permits by giving applicants and cities/counties clearer options when plan checks are taking too long. When a nonresidential permit application is declared complete, the local building department must give an estimated timeframe to finish its review; if that estimate (or the actual delay) would be \"excessive\" — shortened by the bill from 50 days to 30 days — the applicant can ask the jurisdiction to hire a private plan-checking firm, or, if the jurisdiction says none are available, the applicant may hire a qualified private professional (a licensed engineer or architect with recognized plans‑examiner certification) at their own cost. The private reviewer must provide an affidavit and a report, and the city/county must act on that report within 10 business days (or the permit is deemed approved); the bill also requires local governments to publish nonresidential permit fee schedules online and to inspect certain finished nonresidential projects within 10 business days of completion. AB 2418 includes indemnity and liability protections for local agencies while the temporary private review process is used, and its provisions expire on January 1, 2037."
            ),
        TipSheet(
            "California Assembly Committee Unanimously Approves Pawnbroker Tax Bill",
            ["Taxations", "Economy", "Budget", "Business", "Finance"],
            "AB 2641 prevents people who redeem their own pledged items from a pawnbroker from being charged sales tax a second time. Under the bill, when a pawnbroker transfers title back to the person who originally pledged the property (subject to conditions such as the transfer occurring within six months, the person paying the remaining loan balance plus allowable charges, and providing proof they originally paid sales tax), that transfer is not treated as a taxable sale through January 1, 2032. The bill also spells out its goal (to avoid double sales tax), identifies the California Department of Tax and Fee Administration’s utilization estimate as the performance measure, and specifies that the state will not reimburse local governments for any lost sales tax revenue; it takes effect immediately as a tax levy. "
            ),
        
        TipSheet(
            "California Assembly Appropriations Panel Approves Changes to Foster Youth Crisis Program",
            ["Healthcare", "Public Safety", "Budget", "Healthcare"],
            "This bill makes two main changes to California’s Children’s Crisis Continuum Pilot Program, which serves foster youth with acute behavioral health needs. First, it lets a pilot site that cannot run a formal \"crisis residential program\" instead use a functionally equivalent residential treatment component (for example a short-term residential therapeutic or psychiatric residential treatment facility) so long as it provides short-term, intensive, highly individualized crisis stabilization, higher staffing and clinical supports, and is well integrated into the rest of the care continuum; the State Department of Social Services will decide whether a proposed alternative meets those standards and the requirements must be described in licensing program statements. It also allows grant money originally earmarked for a crisis residential program to be used for other parts of the continuum when a comparable component is used, and requires the departments to report by April 1, 2027 on why crisis residential programs have or have not been implemented and what is being done to address barriers. Second, the bill lets the department, in consultation with the Department of Health Care Services and upon a written request, extend a pilot grant’s term for the minimum time needed to finish implementation or closeout and spend remaining funds, but not beyond July 1, 2030."
            ),
        TipSheet(
            "California Assembly Committee Approves Bill to Expand Abortion Provider Eligibility",
            ["Business", "Economy", "Education", "Employment", "Women", "Healthcare"],
            "This bill would broaden who may legally provide abortions in California and set training and safety rules for those providers. It allows clinicians who are authorized by their licensing laws — including nurse practitioners, certified nurse‑midwives, and physician assistants (as well as physicians and osteopathic doctors) — to perform abortions when their professional scope and training permit, and it removes the current limit that such non‑physician clinicians could only do abortions in the first trimester. The bill requires these clinicians to complete specific clinical and classroom training (not just online simulations), to be validated as clinically competent, and, for nurse practitioners and certified nurse‑midwives doing procedural abortions, to have written procedures for consulting, collaborating, referring, and transferring care to a physician for complex cases or emergencies. It also protects people who evaluate trainee competency from civil liability and prevents discipline of physicians who perform abortions in accordance with the law."
            ),
        TipSheet(
            "California Assembly Committee Advances Pedestrian Safety Bill With No Republican Support",
            ["Transportaion", "Enviornment", "Public Safety", "Agriculture"],
            "This bill is designed to make it easier and faster for cities and counties to build pedestrian- and bicycle-focused safety projects and pedestrian-only street areas. It stops local governments from re-opening general public “community input” meetings for bike or walking safety projects once those projects are already included in an approved part of the city or county general plan, and it prevents cancelling such projects after contracts are awarded or construction has begun unless a public meeting shows (by a preponderance of evidence) that cancelling would better serve the public and that the project cannot be funded. If a city or county allows resident petitions for traffic calming, the bill bars requiring more than a simple majority of nearby residents’ signatures (within 1,000 feet). The bill also creates a modernized “Pedestrian Mall Law” that lets cities adopt and improve pedestrian malls (and permit private business uses and improvements) while prohibiting new vehicular parking in those areas, and it makes the establishment or expansion of pedestrian malls exempt from CEQA review (subject to the other procedural, public‑meeting, and workforce requirements that apply to similar transportation exemptions). Finally, the measure declares these rules are a statewide policy that apply to all cities, including charter cities."
            ),
        TipSheet(
            "California Assembly Committee Reviews Food Additive Oversight and Transparency Bill",
            ["Agriculture", "Public Safety", "Healthcare"],
            "AB 2034 tightens oversight of chemicals added to foods and increases ingredient transparency for California consumers. Starting July 1, 2027 the bill requires companies that want to use new or post‑1958 food additives or dietary ingredients to submit a notice with safety information to the California Department of Public Health, which must check the submission and publish the non‑confidential safety data in a searchable public database. The department can use a wide range of health and exposure factors when evaluating additives and must periodically reassess selected additives. The law also requires, by July 1, 2027, manufacturers of packaged foods sold in California to give the department a complete list of products that do not individually name every ingredient on the label and to identify each unnamed ingredient (with common names and chemical identifiers and whether it’s a natural or artificial flavor or color). Small businesses under $1 million in annual sales are exempt, retailers may sell existing inventory through its package date (but no later than July 1, 2030), and the bill does not change federal or state packaging label rules."
            ),
        TipSheet(
            "California Assembly Committee Advances Altered Bill on Mission Bay Park Lands",
            ["San Diego", "Housing", "Finance", "Taxations", "Judiciary", "Economoy", "Enviornment", "Rules"],
            "AB 2525 makes a special rule for Mission Bay Park in San Diego: it allows certain commercially used parcels in the park (those listed in the city’s Mission Bay Park Master Plan, already under lease and already containing commercial/retail/hotel/parking/conference uses as of Jan. 1, 2026) to be treated as “exempt surplus land” — meaning the city can dispose of them under the narrower rules that apply to exempt parcels — but only if the city follows several safeguards. Before disposal the city must publicly find the land is not needed for park or public trust uses, show the disposition won’t harm public recreation or open space, keep lease areas under the city charter’s 25% cap, and give the Department of Housing and Community Development (HCD) written notice at least 30 days beforehand (HCD must tell the city within 30 days if it finds a problem). The law also requires the city to put significant housing funds into a local housing set‑aside account when it disposes of these parcels (either a 10% deposit under certain city-performance conditions or 30% of the sale/fair market or discounted lease value otherwise), and those funds must be used within three years for homes affordable to extremely low, very low, or low‑income households (or else revert to state housing funds). The bill demands that these conditions be recorded as enforceable restrictions on the property and makes violations subject to stiff penalties."
            ),

        TipSheet(
            "Committee Vote California Assembly Appropriations Committee Unanimously Approves 911 Medical Call-Handling Bill",
            ["Communication", "Public Safety"],
            "This bill directs the Governor’s Office of Emergency Services (OES) to include 911 medical call‑handling — specifically the systems that provide prearrival medical instructions to callers — when it does its regular (every two years) review and update of technical and operational standards for public safety communications. In short, it makes sure that the statewide standards review explicitly addresses how 911 centers process medical calls and deliver prearrival instructions (consistent with the existing requirement for agencies to offer those instructions), while clarifying that OES is not being given authority to prescribe the training or the exact content of those medical instructions."
            ),
        TipSheet(
            "California Assembly Committee Passes Homeless Housing Bill Amid GOP Division",
            ["Housing", "Budget", "Economy", "Finance", "Taxations"],
            "This bill creates a new California Direct Access to Supportive Housing (DASH) designation to help people who are homeless or at risk get into supportive housing faster. It lets building sponsors and housing tax credit applicants mark qualifying units as DASH (units restricted to people experiencing or at risk of homelessness and not subject to certain federal HUD subsidies), and requires state housing agencies to adopt faster, simpler tenant screening rules for those units starting July 1, 2027. Those rules generally remove third‑party verification and housing‑history checks and allow an applicant to self‑certify their homelessness status with a signed affidavit (false statements can be grounds for eviction). If a unit listed with the local coordinated entry system remains unfilled after a 180‑day waiting period, sponsors may accept outside referrals under defined conditions (they must notify the coordinated entry system, keep records, ensure the tenant meets eligibility, and return the unit to the coordinated entry process when it next becomes vacant). The bill also directs the housing agencies to review and streamline required documentation by early 2028 and implement changes by July 1, 2028. The overall purpose is to reduce paperwork and delays so supportive housing units are filled more quickly for people in need."
            ),
        TipSheet(
            "California Assembly Passes Bill to Streamline Approval of Small Housing Projects",
            ["San Diego", "Housing", "Budget", "Business", "Economy", "Public Safety"],
            "AB 2601 is designed to speed up and simplify approval of small housing projects and lot splits. It requires cities and counties to allow certain two‑unit homes in single‑family zones and small housing projects (up to 10 units on a subdivided lot) to be processed at the same time as the parcel map or subdivision map (including urban lot splits and, if requested, condominium plans). These projects must be reviewed ministerially (no discretionary hearings), decided within 60 days (or deemed approved), and cannot be blocked by subjective local rules that would effectively prevent the units or make them too small. Local agencies may still condition final permits or certificates of occupancy on recording the approved parcel map, and the law preserves protections such as limits on demolishing rent‑restricted or recently tenant‑occupied housing and allowing denials only for specific, documented public health or safety impacts. The overall effect is to streamline approvals for small infill housing while keeping certain tenant, historic, and safety safeguards."
            ),

        TipSheet(
            "California Assembly Passes Bill Enhancing Homeowner Protections Post-Disaster and Wildfires",
            ["Insurance", "Business", "Economy", "Housing"], 
            "AB 2038 strengthens protections for homeowners after disaster-related total losses and after wildfires. It requires insurers to continue offering to renew a residential property policy for at least three yearly renewals (and for at least 36 months from the date of a total loss) when a home was destroyed by a disaster and rebuilding isn’t finished, and it extends the ban on canceling or refusing to renew policies for properties in ZIP codes within or next to a wildfire perimeter from one year to two years after a state of emergency. The law still lets insurers adjust coverages and premiums after consulting the homeowner, and it allows cancellations or nonrenewals in cases like willful or grossly negligent acts, subsequent unrelated losses, or physical changes that make the property uninsurable. The Department of Forestry and Fire Protection provides the fire perimeter data and the Insurance Commissioner tells insurers which ZIP codes are covered."
        ),
        TipSheet(
            "California Assembly Appropriations Committee Passes Health Investment Disclosure Bill",
            ["Business", "Insurance", "Healthcare"],
            "AB 1929 requires health care service plans and health insurers operating in California to publicly disclose each year their \"material\" investment holdings — meaning significant ownership interests or financial stakes that could affect their finances or create conflicts of interest. The first report, due July 1, 2027, must include holdings for the five years before 2027; thereafter plans and insurers must update the disclosure annually for the prior calendar year. The Department of Managed Health Care and the Department of Insurance must post these disclosures on their websites (and Covered California will flag any noncompliant entities), and any plan or insurer that misses the deadline faces a 30‑day grace period followed by a $1,000 per day civil penalty and must publicly post a notice of noncompliance on its own website until it complies. The bill’s purpose is to increase transparency and public accountability for how organizations that receive public benefits and collect premium dollars invest their financial reserves."
            ),
        TipSheet(
            "California Assembly Committee Passes Health Investment Transparency Bill",
            ["Business", "Insurance", "Healthcare"],
            "AB 1929 requires health care service plans and health insurers operating in California to publicly disclose each year their \"material\" investment holdings — meaning significant ownership interests or financial stakes that could affect their finances or create conflicts of interest. The first report, due July 1, 2027, must include holdings for the five years before 2027; thereafter plans and insurers must update the disclosure annually for the prior calendar year. The Department of Managed Health Care and the Department of Insurance must post these disclosures on their websites (and Covered California will flag any noncompliant entities), and any plan or insurer that misses the deadline faces a 30‑day grace period followed by a $1,000 per day civil penalty and must publicly post a notice of noncompliance on its own website until it complies. The bill’s purpose is to increase transparency and public accountability for how organizations that receive public benefits and collect premium dollars invest their financial reserves."
            ),
        TipSheet(
            "California Committee Advances New Pathway for Graduates to Practice Medicine",
            ["Business", "Ecnomoy", "Education", "Employment", "Medication", "Healthcare"],
            "This bill creates a new \"Physician Graduate\" license that lets people who have graduated from medical school but have not completed a residency practice medicine in California under supervision. To qualify, applicants must have graduated recently (within four years), passed the required medical licensing exams (or equivalents), and, if trained abroad, have ECFMG certification or equivalent; they must also have a written supervising practice agreement with a licensed, board‑certified California physician who maintains an active practice and is not under discipline. The Medical Board would approve the supervision plan, set rules about what physician graduates may and may not do, establish supervision standards (including initial direct supervision, supervisor-to‑trainee ratios, on‑site requirements, and reporting), and require license renewal every three years with continuing education and sponsor evaluations. Physician graduates must disclose their supervised status to patients, and fees would cover program administration. The law is intended to expand supervised, monitored pathways into practice to increase access to care—especially in underserved areas—while maintaining patient safety."
            ),
        TipSheet(
            "California Committee Backs Pilot for Community College Nursing Degrees Amid Labor Division",
            ["Fresno", "Education", "Finance", "Economy", "Employment", "Healthcare"],
            "This bill creates a temporary pilot program to let up to 10 California community college districts offer a Bachelor of Science in Nursing (BSN) degree. The Chancellor’s Office will choose participating districts with an eye toward regional balance, areas with nursing shortages, and communities with persistent poverty, and will give priority to districts that already have (or are close to gaining) national nursing accreditation. Participating colleges must keep their existing associate degree in nursing (ADN) programs, give priority enrollment in the new BSN programs to their own ADN graduates, and limit BSN class sizes to a percentage-based cap. The Chancellor’s Office will help districts seeking accreditation, and the Legislative Analyst’s Office must evaluate the pilot and report back to the Legislature by July 1, 2034. The pilot authority expires on January 1, 2036."
            ),

        TipSheet(
            "California Assembly Committee Passes Nursing Graduate Support Bill Amid GOP Division",
            ["Education", "Finance", "Employment", "Housing", "Healthcare"],
            "This bill creates a New Nursing Graduate Support and Placement Program run by the Department of Health Care Access and Information to help recent graduates of community college associate degree nursing (ADN) programs get into nursing jobs—especially in rural and other underserved areas. The department would award grants to community college ADN programs so they can provide direct financial support to new grads (for things like loan repayment, relocation, housing, childcare, transportation, etc.) and run post-licensure job placement activities (on-campus hiring events, agreements with hospitals, and reporting on employment outcomes). Colleges must work with nursing labor organizations, target hospital sites that meet identified labor standards (for example, union representation, fair pay, good retention practices, and no restrictive repayment contracts), and prioritize applicants with financial need, special circumstances, or residency in underserved areas when grants are limited. The bill also states the Legislature’s intent to strengthen existing nurse training programs under the Song‑Brown Act. Implementation depends on the Legislature providing funding."
            ),

        TipSheet(
            "California Assembly Approves Housing Bill Prohibiting Time Limits in Federally Assisted Housing",
            ["Housing", "Budget", "Economy", "Employment", "Women"],
            "AB 2128 prohibits public housing authorities and other providers of federally assisted housing (for example, public housing, Section 8 vouchers, and project-based rental assistance) from imposing time limits on how long a household may stay or requiring tenants to work, attend school, or perform other \"work activities\" as a condition of getting or keeping housing assistance or the amount of subsidy they receive, unless federal law requires it. The bill still allows providers to offer voluntary employment or job‑training programs, but only if taking part does not affect a household’s eligibility or subsidy and the program’s goals are things like increasing income, savings, homeownership, education, job skills, or employment options. It also preserves applicable federal exceptions, including Moving to Work demonstration rules and existing Family Self‑Sufficiency and public housing community service requirements."
            ),
        TipSheet(
            "California Assembly Committee Advances Bill for Licensing Foreign-Trained Dental Hygienists",
            ["Business", "Economy", "Education", "Employment", "Healthcare"],
            "AB 1952 creates a clear pathway for people who earned a dental hygiene degree from a school that is not accredited in the U.S. to become licensed dental hygienists in California. It requires the Dental Hygiene Board to certify those applicants as eligible to take the American Board of Dental Examiners (ADEX) hygiene exam if they submit an education‑equivalency evaluation, have passed the National Board Dental Hygiene exam and the California law & ethics exam within the last five years, and complete recent (within two years) required coursework — including infection control, a Dental Practice Act course, a board‑approved combined course in soft‑tissue curettage, local anesthesia and nitrous oxide, and current basic life support. The bill also allows applicants who document meeting the other requirements to enroll in that combined clinical course. After an applicant passes the ADEX exam and files an application with required fees, the board must grant a registered dental hygienist license."
            )
        ]

cleanedTitles = ["Represent this sentence for searching relevant passages: " + clean_text(t.title) for t in tipsheets]
cleanedSums = ["Represent this sentence for searching relevant passages: " + clean_text(t.summary) for t in tipsheets]

summary_embs = model.encode(cleanedSums)
title_embs = model.encode(cleanedTitles)
combined_text = ["Represent this sentence for searching relevant passages: "+ f"{cleanedTitles[i]}. {cleanedSums[i]}" for i in range(len(cleanedSums))]
combined_emb = model.encode(combined_text)


query_idx = 15

summary_scores = summary_embs[query_idx] @ summary_embs.T
title_scores = title_embs[query_idx] @ title_embs.T
combined_scores = combined_emb[query_idx] @ combined_emb.T
tag_scores = numpy.array(
    [jaccard_similarity(tipsheets[query_idx].tags, t.tags) for t in tipsheets]
)

print("COMPARING\n" + tipsheets[query_idx].title + "\n--------------------------")

sum_rate = 0.65
title_rate = 0.05
tag_rate = 0.2
com_rate = 0.1

print(f"sum: {sum_rate}, title: {title_rate}, tag: {tag_rate}, combined: {com_rate}")
final = (sum_rate * summary_scores) + (title_rate * title_scores) + (tag_rate * tag_scores) + (com_rate * combined_scores)

# print(final[1], tipsheets[1].title)
# print(final[2], tipsheets[2].title)
# print(final[11], tipsheets[11].title)
for i in range(len(final)):
    if 0.95 > final[i] > 0.55:
        print(i, ":* ", final[i], tipsheets[i].title)
    else:
        print(i, ": ", final[i], tipsheets[i].title)