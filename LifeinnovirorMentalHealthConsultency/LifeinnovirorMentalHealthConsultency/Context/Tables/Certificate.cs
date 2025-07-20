using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class Certificate
    {
        [Key]
        public int CertificateId { get; set; }

        [Required]
        [ForeignKey("Doctor")]
        public int DoctorId { get; set; }
        public Doctor Doctor { get; set; }

        [Required(ErrorMessage = "Certificate URL is required.")]
        [StringLength(255)]
        public string Url { get; set; }

        [StringLength(255)]
        public string Description { get; set; }
    }
}